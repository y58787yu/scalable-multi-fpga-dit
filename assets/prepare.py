"""Build per-role assets for Cosmos-Predict2 fixed-prompt inference."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parameters(model,groups,steps):
    if (model,groups,steps) not in (("2b",1,10),("2b",2,10),("14b",3,5),("14b",3,10)):
        raise ValueError("Use 2b with one/two groups and a 10-step grid, or 14b with three groups and a 5/10-step grid")
    return (2048,28,2) if model=="2b" else (5120,36,4)


def materialize(model,groups,steps,components,state,root,out):
    import numpy as np
    from arithmetic import linear,rms,silu,add,packed_kv
    from inputs import make_inputs,load_context,PROMPTS,SEED
    from model import MatrixView,RankView,GLOBAL_FIELDS,LAYER_FIELDS,digest
    from packing import write_weight,write_cache,json_digest,write_json,link_or_copy

    hidden,layers,ranks=parameters(model,groups,steps)
    own=layers//groups
    heads=hidden//(128*ranks)
    runtime=HERE.parent/"runtime"/model
    profile_path=runtime/(f"tp2_profile_sp{groups}.json" if model=="2b" else "tp4_profile.json")
    profile=read_json(profile_path)
    profile_hash=digest(profile_path)
    config=read_json(HERE/"identities.json")[f"{model}-{groups}-{steps}"]
    preset=config["preset"]
    contexts=load_context()
    root.mkdir()
    times=make_inputs(state,components,steps,hidden,root/"inputs",config["inputs"])
    inputs={"schedule":"inputs/schedule.npz","rope":"inputs/rope.npz","noise_reference":"inputs/noise_reference.npy"}
    files={v:digest(root/v) for v in inputs.values()}
    input_identity=dict(model=profile["model"],preset=preset,prompts=PROMPTS,
                        packed_context_sha256=digest(HERE/"apple_context.npz"),arrays=config["inputs"])
    input_hash=json_digest(input_identity)
    binding=dict(model=profile["model"],official_checkpoint_sha256=config["checkpoint"],input_manifest_sha256=input_hash)
    source_hash=json_digest(binding)
    ctx={}
    for branch,arr in contexts.items():
        full=np.zeros((512,1024),dtype="<f4");full[:len(arr)]=arr
        ctx[branch]=dict(stored_keys=len(arr),logical_keys=512,row_indices=list(range(len(arr))),
                         sha256=hashlib.sha256(full.tobytes()).hexdigest())
    origin=dict(kind="TP4_PRECOMPUTED_ASSETS_V1",model=profile["model"],profile_sha256=profile_hash,
                preset_id=preset,execution=dict(updates=steps,hardware_profile_updates=19),
                model_shape=dict(hidden=hidden,layers=layers,heads=hidden//128,tensor_parallel=ranks,sequence_parallel=groups),
                source_binding=binding,source_binding_sha256=source_hash,contexts=ctx,files=files,ranks={},stages={},
                noise_provenance=dict(kind="SIGMA_SCALED_INITIAL_LATENT",reference_seed=SEED),**inputs)
    kind="C209_TP2_SP12_2B_ASSETS_V1" if model=="2b" else "C209_SP3_REUSE_MANIFEST_V1"
    reuse=dict(kind=kind,model=profile["model"],profile_sha256=profile_hash,source_profile_sha256=profile_hash,
               source_binding_sha256=source_hash,updates=steps,files=files,ranks={},**inputs)
    kind="C209_IP20_2B_REORDERED_ASSETS_V1" if model=="2b" else "C209_IP16_REORDERED_ASSETS_V1"
    packed=dict(kind=kind,model=profile["model"],profile_sha256=profile_hash,files={},ranks={},
                b_layout="C209_B_N640_K128_SHARD128_I16_V1",k_layout="C209_K_HEAD_TOKEN128_I16_V1",
                patch_input_policy="FIXED_SAMPLER_A_ZERO64_79_EXISTING_F3_K64_V1",sampler_sha256=digest(runtime/"c206_sampler.py"))
    def register(entry):
        path,actual=entry["file"],entry["sha256"]
        if actual!=config["packed"].get(path):
            raise ValueError(f"Numerical identity mismatch: {path}")
        packed["files"][path]=actual

    global_weights=MatrixView(state)
    name="weights/global.bin"
    meta,entry,raw_files=write_weight(root,name,GLOBAL_FIELDS,global_weights)
    files.update(raw_files);register(entry)
    origin["global"]=reuse["global_file"]=name
    reuse["global_layout"]=meta
    packed["global"]=entry
    def gemm(w,key,x):
        return linear(x,w.get(key+".q"),w.get(key+".s"))
    table=np.array([int(s,16) for s in (HERE/"silu_u32.hex").read_text().split()],dtype="<u4")
    timeglobal=gemm(global_weights,"time2",silu(gemm(global_weights,"time1",times),table))
    timesilu=silu(rms(times,global_weights.get("time_gamma")),table)
    final=add(gemm(global_weights,"final2",gemm(global_weights,"final1",timesilu)),timeglobal[:,:2*hidden])
    del global_weights
    for r in range(ranks):
        rank=str(r)
        origin["ranks"][rank]={"layers":{}}
        reuse["ranks"][rank]={"layers":{},"caches":{}}
        packed["ranks"][rank]={"layers":{},"caches":{}}
    for g in range(groups):
        common=np.empty((steps,own,3,3*hidden),dtype="<f4")
        cross={r:{c:{} for c in contexts} for r in range(ranks)}
        for local in range(own):
            layer=own*g+local
            full=MatrixView(state,layer)
            for idx,op in enumerate(("self_attn","cross_attn","mlp")):
                prefix="adaln_modulation_"+op
                common[:,local,idx,:]=add(gemm(full,prefix+".2.weight",gemm(full,prefix+".1.weight",timesilu)),timeglobal)
            for r in range(ranks):
                rank=str(r);w=RankView(full,r,ranks)
                name=f"weights/rank{r}/layer{layer:02}.bin"
                meta,entry,raw_files=write_weight(root,name,LAYER_FIELDS,w)
                files.update(raw_files);register(entry)
                origin["ranks"][rank]["layers"][str(layer)]=name
                reuse["ranks"][rank]["layers"][str(layer)]=dict(file=name,owner=g,layout=meta,sha256=entry["source_sha256"])
                packed["ranks"][rank]["layers"][str(layer)]=dict(entry,owner=g)
                for c,context in contexts.items():
                    k=gemm(w,"cross_attn.k_proj.weight",context)
                    v=gemm(w,"cross_attn.v_proj.weight",context)
                    k=rms(k.reshape(-1,128),w.get("cross_attn.k_norm.weight")).reshape(len(context),heads*128)
                    values=packed_kv(k,v,heads)
                    cross[r][c].update({f"layer{layer:02}.{key}":value for key,value in values.items()})
            del w,full
            print(f"Packed layer {layer+1}/{layers}",flush=True)
        group=str(g)
        stages=origin["stages"].setdefault(group,{"ranks":{}})["ranks"]
        for r in range(ranks):
            rank=str(r)
            original_entry=stages[rank]={}
            new_entry={}
            for key in ("common","positive","negative"):
                data={"modulation":common,"final":final} if key=="common" else cross[r][key]
                name=f"cache/stage{g}_rank{r}_{key}.bin"
                src,k=write_cache(root,name,data,heads,None if key=="common" else len(contexts[key]))
                register(k);files[name]=src["sha256"]
                original_entry[key]=src;new_entry[key]=k
            reuse["ranks"][rank]["caches"][group]=original_entry
            packed["ranks"][rank]["caches"][group]=new_entry
    if packed["files"]!=config["packed"]:
        raise ValueError("Incomplete packed weight/cache identity inventory")
    proof="precompute_verification.json"
    proof_hash=write_json(root/proof,dict(status="PASS_ALL_PRECOMPUTE_ARRAYS_BIT_EXACT_NOT_FULL_MODEL",
                                         source_binding_sha256=source_hash,packed_sha256=json_digest(packed["files"])))
    files[proof]=proof_hash
    origin["numerical_validation"]=dict(file=proof,sha256=proof_hash)
    origin["status"]="VERIFIED_PRECOMPUTE_ACTUAL_WEIGHTS_NOT_FULL_MODEL"
    main_hash=write_json(root/"tp4_assets.json",origin)
    reuse.update(source_manifest_sha256=main_hash,source_numerical_validation=origin["numerical_validation"])
    sp_hash=write_json(root/"sp3_assets.json",reuse)
    packed.update(source_manifest_sha256=main_hash,source_reuse_manifest_sha256=sp_hash)
    hot_hash=write_json(root/"integrated_assets.json",packed)
    release=dict(kind="COSMOS_PUBLIC_ASSETS_V1",model=profile["model"],groups=groups,updates=steps,
                 profile_sha256=profile_hash,asset_manifest_sha256=main_hash,source_binding_sha256=source_hash,
                 input_manifest_sha256=input_hash,reuse_manifest_sha256=sp_hash,packed_manifest_sha256=hot_hash,
                 payload_sha256=json_digest(packed["files"]),preset=preset)
    write_json(root/"release_binding.json",release)
    base={"release_binding.json","tp4_assets.json","sp3_assets.json","integrated_assets.json",
          proof,packed["global"]["file"],"weights/global.layout.json",*inputs.values()}
    if model=="2b":
        cfg=dict(kind="C209_2B_FIXED_ASSET_BINDING_V1",model=profile["model"],hardware_id=profile["hardware_id"],
                 hidden=hidden,layers=layers,heads=hidden//128,tensor_parallel=ranks,sequence_parallel=groups,
                 **PROMPTS,stored_keys=[16,11],logical_keys=512,asset_updates=steps,preset=preset,
                 asset_manifest_sha256=main_hash,input_manifest_sha256=input_hash,source_binding_sha256=source_hash)
        write_json(root/"fixed_2b_binding.json",cfg)
        base.add("fixed_2b_binding.json")
    out.mkdir()
    for g in range(groups):
        for r in range(ranks):
            dest=out/f"node{g*ranks+r+1}"
            names=set(base)
            rank=packed["ranks"][str(r)]
            for l,item in rank["layers"].items():
                names.add(Path(item["source_file"]).with_suffix(".layout.json").as_posix())
                if item["owner"]==g: names.add(item["file"])
            names.update(entry["file"] for stage in rank["caches"].values() for entry in stage.values())
            for name in sorted(names):
                link_or_copy(root/name,dest/name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=("2b","14b"),required=True)
    parser.add_argument("--groups",type=int,required=True)
    parser.add_argument("--steps",type=int,choices=(5,10),default=10)
    parser.add_argument("--weights",type=Path,required=True,help="Official BF16 model.pt")
    parser.add_argument("--components",type=Path,required=True,help="Shared Cosmos Diffusers tree containing scheduler/")
    parser.add_argument("--out",type=Path,required=True,help="New directory for per-role assets")
    parser.add_argument("--threads",type=int,default=4,help="CPU threads used during packing")
    args=parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    parameters(args.model,args.groups,args.steps)
    target=args.out.resolve()
    if target.exists(): parser.error("--out must be a new directory")
    target.parent.mkdir(parents=True,exist_ok=True)
    import os
    for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[name]=str(args.threads)
    from model import load
    import torch
    torch.set_num_threads(args.threads)
    cfg=read_json(HERE/"identities.json")[f"{args.model}-{args.groups}-{args.steps}"]
    state=load(args.weights,cfg["checkpoint"])
    with TemporaryDirectory(prefix=".cosmos-assets-",dir=target.parent) as td:
        temp=Path(td)
        materialize(args.model,args.groups,args.steps,args.components,state,temp/"packed",temp/"roles")
        if target.exists(): raise FileExistsError(target)
        (temp/"roles").rename(target)
    print(target)
if __name__=="__main__":
    main()
