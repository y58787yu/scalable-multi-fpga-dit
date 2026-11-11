"""Fixed prompt conditioning, rotary positions, reference noise and Karras grid."""
from pathlib import Path
import hashlib
import math
import numpy as np

PROMPTS = {"positive": "A red apple on a wooden table, warm sunlight, detailed photograph.",
           "negative": "blurry, distorted, low quality, noisy"}
SEED = 20261006


def load_context():
    with np.load(Path(__file__).with_name("apple_context.npz"), allow_pickle=False) as f:
        return {condition: f[condition].astype("<f4") for condition in PROMPTS}


def timesteps(times, hidden):
    import torch
    exponent = -math.log(10000) * torch.arange(hidden//2,dtype=torch.float32) / (hidden//2)
    phase = torch.from_numpy(times)[:,None] * torch.exp(exponent)[None,:]
    return torch.cat((phase.cos(),phase.sin()),dim=-1)[:,None,:].numpy().astype("<f4")


def schedule(directory, updates, hidden):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    sampler=FlowMatchEulerDiscreteScheduler.from_pretrained(directory, local_files_only=True)
    if not (sampler.config.use_karras_sigmas and sampler.config.final_sigmas_type=="sigma_min"
            and float(sampler.config.sigma_max)==80 and float(sampler.config.sigma_min)==.002
            and not sampler.config.stochastic_sampling):
        raise ValueError("Expected Cosmos Predict2 deterministic Karras scheduler")
    sampler.set_timesteps(sigmas=torch.linspace(0,1,updates+1,dtype=torch.float64).numpy(),device="cpu")
    sampler.sigmas[-1]=sampler.sigmas[-2]
    sigmas=sampler.sigmas.numpy().astype("<f4")
    idx=np.flatnonzero(sigmas[1:]!=sigmas[:-1]).astype("<i4")
    if len(idx)!=updates:
        raise ValueError("Schedule did not produce the requested number of updates")
    times=np.array([float(sampler.sigmas[i]/(sampler.sigmas[i]+1)) for i in idx],dtype="<f4")
    return dict(sigmas=sigmas,timestamps=sampler.timesteps.numpy().astype("<f4"),indices=idx,
                times=times,coefficients=np.array([1-float(t) for t in times],dtype="<f4"),
                deltas=np.array([float(sampler.sigmas[i+1]-sampler.sigmas[i]) for i in idx],dtype="<f4"),
                sin_times=timesteps(times,hidden))


def rope(state):
    import torch
    seq=state["net.pos_embedder.seq"].float()
    spatial=state["net.pos_embedder.dim_spatial_range"].float()
    temporal=state["net.pos_embedder.dim_temporal_range"].float()
    dim=2*len(spatial)
    spatial_rate=1.0/((10000.0 * (4.0**(dim/(dim-2))))**spatial)
    temporal_rate=1.0/(10000.0**temporal)
    positions=torch.outer(seq[:64],spatial_rate)
    t=torch.outer(seq[:1],temporal_rate).reshape(1,1,-1).expand(64,64,-1)
    y=positions[:,None,:].expand(64,64,-1)
    x=positions[None,:,:].expand(64,64,-1)
    angles=torch.cat([t,y,x]*2,dim=-1).reshape(4096,128)
    return {"cos":angles.cos().numpy(),"sin":angles.sin().numpy()}


def make_inputs(state, components, updates, hidden, target, expected):
    import torch
    target.mkdir()
    grid=schedule(components/"scheduler",updates,hidden)
    rotary=rope(state)
    noise=(torch.randn((1,16,1,128,128),generator=torch.Generator(device="cpu").manual_seed(SEED),
                       dtype=torch.float32)*float(grid["sigmas"][0])).numpy()
    for name,arrays in (("rope",rotary),("schedule",grid),("noise_reference",{"array":noise})):
        actual={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
        if actual!=expected[name]:
            raise ValueError(f"{name} does not match the fixed configuration. Install assets/requirements.txt.")
    np.savez(target/"rope.npz",**rotary)
    np.savez(target/"schedule.npz",**grid)
    np.save(target/"noise_reference.npy",noise,allow_pickle=False)
    return grid["sin_times"].reshape(updates,hidden)
