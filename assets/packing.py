"""Write panel-major weights, head-major keys and aligned cache blocks."""
import hashlib
import json
from pathlib import Path
import numpy as np

from model import digest


def json_bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=True)+"\n").encode("utf8")


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(json_bytes(value))
    return digest(path)


def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def aligned(n):
    return (n+63)//64*64


def weight_panels(value):
    n,k=value.shape
    effective=64 if k==80 else k
    for block in range(0,n,640):
        for inner in range(0,effective,128):
            for offset in range(0,min(640,n-block),128):
                yield np.ascontiguousarray(value[block+offset:min(n,block+offset+128),
                                                inner:min(effective,inner+128)]).tobytes()


def write_weight(root, file, fields, view):
    destination="integrated/"+Path(file).with_suffix(".panels.bin").as_posix()
    target=root/destination
    target.parent.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256()
    source_offset=out_offset=0
    source_arrays,packed_arrays={},{}
    with target.open("xb") as fp:
        for key in fields:
            value=np.ascontiguousarray(view.get(key))
            physical=list(value.shape)
            spos,ppos=aligned(source_offset),aligned(out_offset)
            source_hash.update(bytes(spos-source_offset))
            source_hash.update(memoryview(value).cast("B"))
            fp.write(bytes(ppos-out_offset))
            source_arrays[key]=dict(offset=spos,bytes=value.nbytes,shape=list(value.shape),dtype=value.dtype.str)
            source_offset=spos+value.nbytes
            entry=dict(dtype=value.dtype.str,shape=physical,offset=ppos,source_offset=ppos)
            if key.endswith(".q"):
                entry.update(matrix_layout="C209_B_N640_K128_SHARD128_I16_V1",logical_shape=list(value.shape))
                physical[1]=64 if physical[1]==80 else physical[1]
                for data in weight_panels(value): fp.write(data)
            else:
                fp.write(memoryview(value).cast("B"))
            nbytes=int(np.prod(physical))*value.dtype.itemsize
            entry["bytes"]=nbytes
            packed_arrays[key]=entry
            out_offset=ppos+nbytes
        fp.write(bytes(aligned(out_offset)-out_offset))
    source_hash.update(bytes(aligned(source_offset)-source_offset))
    layout=dict(bytes=aligned(source_offset),arrays=source_arrays)
    layout_file=str(Path(file).with_suffix(".layout.json").as_posix())
    layout_hash=write_json(root/layout_file,layout)
    compact=dict(bytes=layout["bytes"],source_bytes=layout["bytes"],
                 arrays={k:dict(v,source_offset=v["offset"]) for k,v in source_arrays.items()})
    packed=dict(file=destination,sha256=digest(target),source_file=file,source_sha256=source_hash.hexdigest(),
                layout=dict(bytes=aligned(out_offset),source_bytes=aligned(out_offset),arrays=packed_arrays))
    return compact,packed,{layout_file:layout_hash,file:source_hash.hexdigest()}


def write_cache(root,name,arrays,heads,stored_keys=None):
    destination="integrated/"+name
    target=root/destination
    target.parent.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256()
    meta={}
    pos=0
    with target.open("xb") as fp:
        for key,data in arrays.items():
            a=np.ascontiguousarray(data)
            begin=aligned(pos)
            pad=bytes(begin-pos)
            source_hash.update(pad)
            source_hash.update(memoryview(a).cast("B"))
            fp.write(pad)
            ktype=key.rsplit(".",1)[-1]
            raw=ktype in ("k","v")
            meta[key]=dict(offset=begin,bytes=a.nbytes,shape=[a.nbytes] if raw else list(a.shape),
                           dtype="|u1" if raw else a.dtype.str)
            if ktype=="k":
                a=np.ascontiguousarray(a.reshape(stored_keys,heads,128).transpose(1,0,2))
            fp.write(memoryview(a).cast("B"))
            pos=begin+a.nbytes
        end=aligned(pos)
        fp.write(bytes(end-pos))
        source_hash.update(bytes(end-pos))
    entry=dict(file=name,bytes=end,sha256=source_hash.hexdigest(),arrays=meta)
    if stored_keys is not None:
        entry["stored_keys"]=stored_keys
    packed=dict(entry,file=destination,sha256=digest(target),k_layout="C209_K_HEAD_TOKEN128_I16_V1",
                source_file=name,source_sha256=entry["sha256"])
    return entry,packed


def link_or_copy(source,target):
    import os
    import shutil
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        os.link(source,target)
    except OSError:
        shutil.copyfile(source,target)
