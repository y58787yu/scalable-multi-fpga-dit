"""Explicit CMA allocation plan; cold twelve-layer files remain on their owner."""
import math
from c209_tensor_layout import PackedKVLayout
from sp3_layout import ROWS
from sp3_overlap_schedule import SCRATCH_NAMES

def guarded(n):return (n+256+4095)//4096*4096

def memory_plan(role,layer_bytes,cache_bytes,global_bytes,*,budget_bytes=880*1024**2,updates=10):
    n=role.rows;d=2048;w=1024
    shapes={'state0':((n,d),'<f4'),'state1':((n,d),'<f4'),'quant':((n,2048),'<i2'),
        'sa':((n,),'<f4'),'q':((n,w),'<f2'),'k':((n,w),'<f2'),'v':((n,w),'<f2'),
        'attn':((n,w),'<f2'),'ffn':((n,4096),'<f2'),'ffn_scales':((n,),'<f4'),
        'input':((n,80),'<f4'),'prediction':((n,64),'<f4'),
        'positive_full':((4096,64),'<f4'),'negative_full':((4096,64),'<f4'),
        'latent':((1,16,1,128,128),'<f4'),
        'cos':((n,128),'<f4'),'sin':((n,128),'<f4')}
    # The original scratch allocation belongs to the negative branch. The
    # positive copy prevents a producer from overwriting a live KV/TP reader.
    for name in SCRATCH_NAMES:shapes['positive_'+name]=shapes[name]
    for condition in ('positive','negative'):
        for operator in range(3):shapes[f'{condition}_gate{operator}']=((2048,),'<f4')
    for kind in ('partial','sum'):
        for slot in range(4):shapes[kind+str(slot)]=((256,d),'<f4')
    sizes={k:math.prod(shape)*(2 if dtype.endswith('2') else 4) for k,(shape,dtype) in shapes.items()}
    local_kv=0;full_kv=PackedKVLayout(4096,8).guarded_bytes
    hot=2*guarded(max(layer_bytes));fixed=sum(map(guarded,sizes.values()))+local_kv+2*full_kv+guarded(global_bytes)
    cache=sum(guarded(v) for v in cache_bytes);transport=16*1024**2;reserve=32*1024**2;descriptor=guarded(16384)
    if type(updates)is not int or not 1<=updates<=19:raise ValueError('Actual coefficient schedule extent')
    # Scale records have a 64-byte stride; only the first eight U32 words are
    # the SCL1 ABI. They are derived from fixed coefficients, not seed data.
    policy=guarded(updates*28*3*64)+guarded(updates*64)
    derived=guarded(updates*28*3*2048*4)+guarded(updates*2048*4)+policy
    required=fixed+hot+cache+transport+reserve+descriptor+derived
    if required>budget_bytes:raise MemoryError(f'SP3 CMA {required} exceeds {budget_bytes}')
    return dict(shapes=shapes,tensor_bytes=sizes,local_kv_staging_bytes=local_kv,global_kv_bytes=2*full_kv,
        hot_weight_bytes=hot,cache_bytes=cache,global_bytes=global_bytes,required_cma_bytes=required,
        budget_bytes=budget_bytes,transport_reserve_bytes=transport,engineering_reserve_bytes=reserve,
        derived_coefficient_bytes=derived,scale_policy_bytes=policy,local_to_global_kv_copy=False,ffn_quant_ddr_materialized=False,
        cfg_workspaces=2,partial_slots=4,partial_slots_per_branch=2,
        cold_owner_layers=list(role.owned_layers),cold_owner_bytes=sum(layer_bytes[i] for i in role.owned_layers),
        remote_layers_resident_on_sd=False,simultaneously_live_weight_slots=2)
