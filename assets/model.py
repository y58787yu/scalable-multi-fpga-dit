"""Read official BF16 matrices by memory map and partition only after quantization."""
import hashlib
from pathlib import Path
import numpy as np

GLOBAL = {"patch":"x_embedder.proj.1.weight", "time1":"t_embedder.1.linear_1.weight",
          "time2":"t_embedder.1.linear_2.weight", "final1":"final_layer.adaln_modulation.1.weight",
          "final2":"final_layer.adaln_modulation.2.weight", "final":"final_layer.linear.weight",
          "time_gamma":"t_embedding_norm.weight"}
GLOBAL_FIELDS = ("patch.q", "patch.s", "final.q", "final.s")
LINEARS = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
           "self_attn.output_proj.weight", "cross_attn.q_proj.weight", "cross_attn.output_proj.weight",
           "mlp.layer1.weight", "mlp.layer2.weight")
LAYER_FIELDS = tuple(f"{key}.{part}" for key in LINEARS for part in ("q","s")) + (
    "self_attn.q_norm.weight","self_attn.k_norm.weight","cross_attn.q_norm.weight")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def quantize_weight(tensor):
    import torch
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16:
        raise ValueError("Expected an official BF16 matrix")
    n, k = tensor.shape
    weights = np.zeros((n, (k+15)//16*16), dtype="<i2")
    scales = np.empty(n,dtype="<f4")
    for i in range(0,n,128):
        rows = tensor[i:i+128].float().numpy()
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite checkpoint weight")
        scale = np.max(np.abs(rows),axis=1) / np.float32(32760)
        scale[scale==0] = 1
        scales[i:i+len(rows)] = scale
        scaled = rows / scale[:,None]
        weights[i:i+len(rows),:k] = np.clip(
            np.where(scaled>=0,np.floor(scaled+.5),np.ceil(scaled-.5)), -32760,32760).astype("<i2")
    return weights,scales


class MatrixView:
    def __init__(self, state, layer=None):
        self.state = state
        self.layer = layer
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            matrix = name.endswith((".q",".s"))
            base = name[:-2] if matrix else name
            key = "net." + (GLOBAL[base] if self.layer is None else f"blocks.{self.layer}.{base}")
            tensor = self.state[key]
            if matrix:
                self.cache[base+".q"],self.cache[base+".s"] = quantize_weight(tensor)
            else:
                self.cache[base] = tensor.float().numpy()
        return self.cache[name]


class RankView:
    def __init__(self, full, rank, ranks):
        self.full,self.rank,self.ranks = full,rank,ranks

    def get(self, name):
        value = self.full.get(name)
        if not name.endswith((".q",".s")):
            return value
        op, suffix = name.rsplit(".",1)
        if op.endswith(("q_proj.weight","k_proj.weight","v_proj.weight","mlp.layer1.weight")):
            width = value.shape[0]//self.ranks
            return value[self.rank*width:(self.rank+1)*width]
        if suffix=="q" and op.endswith(("output_proj.weight","mlp.layer2.weight")):
            width = value.shape[1]//self.ranks
            return value[:,self.rank*width:(self.rank+1)*width]
        return value


def load(weights, expected):
    import torch
    if digest(weights)!=expected:
        raise ValueError("Checkpoint differs from this model's official BF16 model.pt")
    return torch.load(weights, mmap=True, weights_only=True, map_location="cpu")
