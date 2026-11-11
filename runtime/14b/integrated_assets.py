"""Immutable IP16 asset overlay: byte reordering only, no new precision."""
import hashlib
import json
from pathlib import Path
import numpy as np

B_LAYOUT='C209_B_N640_K128_SHARD128_I16_V1'
K_LAYOUT='C209_K_HEAD_TOKEN128_I16_V1'
PATCH_INPUT_POLICY='FIXED_SAMPLER_A_ZERO64_79_EXISTING_F3_K64_V1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def panel_chunks(value):
    """The exact loader walk; patch K64 follows the fixed sampler A-zero contract."""
    if value.dtype!=np.dtype('<i2') or value.ndim!=2:raise ValueError('I16 matrix required')
    n,k=value.shape
    effective=64 if k==80 else k
    if n not in (64,1280,5120) or effective%128 and effective!=64:raise ValueError('Fixed panel geometry')
    for nb in range(0,n,640):
        for kb in range(0,effective,128):
            kk=min(128,effective-kb)
            for shard in range(5):
                first=nb+shard*128
                if first>=n:break
                yield np.ascontiguousarray(value[first:min(first+128,n),kb:kb+kk]).tobytes()


def overlay(folder,source_manifest,source_reuse):
    p=Path(folder)/'integrated_assets.json'
    d=json.loads(p.read_text(encoding='utf-8'))
    if (d.get('kind')!='C209_IP16_REORDERED_ASSETS_V1' or d.get('b_layout')!=B_LAYOUT or d.get('k_layout')!=K_LAYOUT
        or d.get('patch_input_policy')!=PATCH_INPUT_POLICY
        or d.get('sampler_sha256')!=digest(Path(__file__).with_name('c206_sampler.py'))
        or d.get('source_manifest_sha256')!=source_manifest or d.get('source_reuse_manifest_sha256')!=source_reuse):
        raise ValueError('New image requires its exact packed B/head-major K overlay')
    return d


def validate_layer(original,new):
    if set(original['arrays'])!=set(new['arrays']):raise ValueError('Packed hot array names changed')
    end=0
    for name,old in original['arrays'].items():
        item=new['arrays'][name];shape=old['shape'];physical=list(shape)
        if name.endswith('.q'):
            if item.get('matrix_layout')!=B_LAYOUT or item.get('logical_shape')!=shape:raise ValueError('Packed matrix binding')
            if shape[1]==80:physical[1]=64
        if item['shape']!=physical or item['dtype']!=old['dtype'] or item['offset']%64 or item['offset']<end:
            raise ValueError('Packed array extent/type changed')
        size=int(np.prod(physical))*np.dtype(old['dtype']).itemsize
        if item['bytes']!=size or item['source_offset']!=item['offset']:raise ValueError('Packed raw extent')
        end=item['offset']+size
    if new['bytes']!=(end+63)//64*64:raise ValueError('Packed layer allocation extent')


def provisioning_files(source,packed,role):
    """One owned 12-layer file set; remote layers remain metadata only."""
    rank=source['ranks'][str(role.rank)];new=packed['ranks'][str(role.rank)]
    files={'tp4_assets.json','sp3_assets.json','integrated_assets.json',source['rope'],source['schedule'],source['noise_reference'],source['source_numerical_validation']['file'],packed['global']['file']}
    files.update(new['layers'][str(i)]['file'] for i in role.owned_layers)
    files.update(Path(rank['layers'][str(i)]['file']).with_suffix('.layout.json').as_posix() for i in range(36))
    files.add(Path(source['global_file']).with_suffix('.layout.json').as_posix())
    files.update(c['file'] for g in new['caches'].values() for c in g.values())
    return sorted(files)
