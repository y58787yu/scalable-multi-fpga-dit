"""Finite TP2 x SP1/SP2 geometry; permanent weight ownership is separate."""
import json
import os
import hashlib
import struct
from pathlib import Path
from dataclasses import dataclass
from c209_program import validate_job as typed_job
from c209_job_ranges import validate_job_ranges
from c209_tensor_layout import ADDRESS_FIELDS

def load_profile(path=None):
    if path is None:path=os.environ.get('C209_TP2_PROFILE')
    if path is None:
        here=Path(__file__).resolve().parent
        path=next((p for p in (here/'tp4_profile.json',here.parent/'config/tp4_profile.json') if p.is_file()),None)
    if path is None:raise FileNotFoundError('The deployed SP3 profile is missing')
    raw=Path(path).read_bytes();p=json.loads(raw)
    if (p['kind']!='C209_2B_TP2_SP12_FIXED_APPLE_PROFILE_V1' or p['pipeline_stages']!=1 or
        p['hardware_id']!='0xC2092B10' or p['runtime_id']!='0xC2092B11' or p['ip_version']!='20.0' or
        p.get('compute_hardware_id')!='0xC2092B00' or
        p.get('shell_variant')!='C209_2B_TP2_SP12_TWO_PORT_V1' or
        p.get('sfp_ports')!=2 or p.get('present_ports')!=[1,2] or
        p.get('active_ports')!=([1] if p['sequence_parallel']==1 else [1,2]) or
        p['physical_mac_shards']!=5 or p['mac_lanes_per_shard']!=256 or p['physical_mac_lanes']!=1280 or
        p['tensor_parallel']*p['sequence_parallel']!=p['boards'] or
        p['hidden']!=p['heads']*p['head_dim'] or
        p['local_width']*p['tensor_parallel']!=p['hidden'] or
        p['local_heads']*p['tensor_parallel']!=p['heads'] or
        p['layers_per_weight_group']*p['weight_groups']!=p['layers'] or
        p['tensor_parallel']!=2 or p['sequence_parallel'] not in (1,2) or
        p['hidden']!=2048 or p['layers']!=28 or p['local_heads']!=8 or p['local_ffn']!=4096 or
        p['sequence_rows']!=[4096//p['sequence_parallel']]*p['sequence_parallel'] or
        p['sequence_offsets']!=[i*(4096//p['sequence_parallel']) for i in range(p['sequence_parallel'])] or
        sum(p['sequence_rows'])!=p['tokens'] or p['compute_rows']!=[2048,4096] or p['partial_rows']!=[128,256]):
        raise ValueError('Inconsistent fixed TP2 x SP1/SP2 geometry')
    if p.get('transport')!={
        'tp_port':1,'tp_peer':'rank_xor_1','tp_rounds':1,
        'tp_operation':'FUSED_SUM_GATE_RESIDUAL','sp_ports':[2],
        'sp1_remote_members':0,'sp2_peer':'group_xor_1',
        'sp2_wiring':'group0.P2-group1.P2','absent_ports':[0,3],
        'kv_dispatch':'ORDERED_SINGLE_LANE_FULL_DUPLEX_V1'}:
        raise ValueError('Two-port shell transport differs from the finite contract')
    p['profile_sha256']=hashlib.sha256(raw).hexdigest()
    return p

PROFILE=load_profile()
HARDWARE_PROFILE=load_profile(Path(__file__).resolve().parent/'tp2_profile_sp2.json')
_SP1_PROFILE=load_profile(Path(__file__).resolve().parent/'tp2_profile_sp1.json')
SUPPORTED_PROFILE_BINDINGS={name:dict(profile_sha256=p['profile_sha256'],boards=p['boards'],tensor_parallel=2,sequence_parallel=p['sequence_parallel'],sequence_rows=p['sequence_rows'],sequence_offsets=p['sequence_offsets'],layers_per_weight_group=p['layers_per_weight_group'],present_ports=p['present_ports'],active_ports=p['active_ports']) for name,p in (('SP1',_SP1_PROFILE),('SP2',HARDWARE_PROFILE))}
if PROFILE['profile_sha256']!=SUPPORTED_PROFILE_BINDINGS['SP'+str(PROFILE['sequence_parallel'])]['profile_sha256']:
    raise ValueError('Selected profile is not one of the two packaged finite profiles')

BIT_ID=int(PROFILE['hardware_id'],16)
COMPUTE_BIT_ID=int(PROFILE['compute_hardware_id'],16)
RUNTIME_ID=int(PROFILE['runtime_id'],16)
PRESENT_PORTS=tuple(PROFILE['present_ports'])
ACTIVE_PORTS=tuple(PROFILE['active_ports'])
EPOCHS_PER_GENERATION=1<<20
MAX_GENERATION=(0xffffffff-(EPOCHS_PER_GENERATION-1))//EPOCHS_PER_GENERATION

def validate_generation(value):
    if type(value)is not int or not 1<=value<=MAX_GENERATION:
        raise ValueError(f'generation must be a monotonically increasing counter1..{MAX_GENERATION}')
    return value

@dataclass(frozen=True)
class Role:
    stage:int
    rank:int
    def __post_init__(self):
        if type(self.stage)is not int or not 0<=self.stage<PROFILE['sequence_parallel']:raise ValueError('sequence group')
        if type(self.rank)is not int or not 0<=self.rank<PROFILE['tensor_parallel']:raise ValueError('rank')
    @property
    def group(self):return self.stage
    @property
    def node(self):return self.rank+1
    @property
    def physical_node(self):return self.group*PROFILE['tensor_parallel']+self.node
    @property
    def layers(self):return range(PROFILE['layers'])
    @property
    def owned_layers(self):
        first=self.group*PROFILE['layers_per_weight_group']
        return range(first,first+PROFILE['layers_per_weight_group'])
    @property
    def rows(self):return PROFILE['sequence_rows'][self.group]
    @property
    def token_offset(self):return PROFILE['sequence_offsets'][self.group]

def condition_id(condition):
    if condition not in ('positive','negative'):raise ValueError('Unknown CFG context')
    return ('positive','negative').index(condition)

def validate_job(job):
    j=typed_job(job);p=PROFILE
    if j['stage'] not in (1,2,4,7,13,14,15,16,17):raise ValueError('Operation is not in the IP12 production call graph')
    D,W,F,H=p['hidden'],p['local_width'],p['local_ffn'],p['local_heads']
    s,m,n,k,la,lb,lc,f=j['stage'],j['rows'],j['cols'],j['inner'],j['lda'],j['ldb'],j['ldc'],j['flags']
    local=m in p['compute_rows'];partial=m in p['partial_rows'];allowed=False
    packed=bool(f&64) if s in (1,15,16) else False
    base_flags=f&~64 if packed else f
    if s==1:
        f=base_flags
        allowed=(local and (n,k,la,lb,lc,f) in ((D,80,80,80,D,0),(W,D,D,D,W,0),(F,D,D,D,F,0),(64,D,D,D,64,0))) or (partial and (n,k,la,lb,lc,f) in ((D,W,W,W,D,8),(D,F,F,F,D,8)))
    elif s==2:allowed=local and not f and (n,la,lc) in ((68,80,80),(W,W,W),(F,F,F))
    elif s==10:allowed=local and not f and (n,la,lc)==(D,D,D)
    elif s==4:allowed=m in [r*H for r in p['compute_rows']] and not f and (n,la,lc)==(128,128,128)
    elif s==6:allowed=local and not f and (n,la,lc,j['heads'])==(W,W,W,H)
    elif s==5:allowed=local and not f and j['mode']==4 and (n,la,lc)==(F,F,F)
    elif s==12:allowed=local and not f and (n,j['k_count'],la,lb,lc,j['heads'])==(m,m,W,W,W,H)
    elif s==7:allowed=local and (la,lb,lc,j['heads'])==(W,W,W,H) and ((f,n,j['k_count'])==(128,p['tokens'],p['tokens']) or (f==160 and n==p['logical_cross_keys'] and j['k_count'] in (16,11)))
    elif s==13:allowed=not f and (j['k_start'],m) in tuple(zip(p['sequence_offsets'],p['sequence_rows'])) and (n,j['k_count'],la,lb,lc,j['heads'])==(p['tokens'],m,W,W,W,H)
    elif s==14:allowed=local and not f and (n,la,lc)==(D,D,D)
    elif s==15:allowed=local and base_flags==0 and (n,k,la,lb,lc)==(F,D,D,D,F)
    elif s==16:allowed=partial and base_flags==8 and (n,k,la,lb,lc)==(D,F,F,F,D)
    elif s==17:allowed=local and not f and (n,la,lc,j['heads'])==(W,W,W,H)
    if s in (1,15,16) and not packed:allowed=False
    if not allowed:raise ValueError('Descriptor outside SP3 fixed shape contract')
    if ((s!=13 and j['k_start']) or j['factor'] or (s not in (1,15,16) and k) or j['mode']!=(4 if s==5 else 0)
        or j['heads']!=(H if s in (6,7,12,13,17) else 0) or (s not in (7,12,13) and j['k_count'])
        or struct.pack('<f',j['epsilon']) not in (struct.pack('<f',0),struct.pack('<f',1e-6))):raise ValueError('Unsupported fixed flags/window')
    expected={2:(2 if n==68 else 1,0,3,2,0,0,0),10:(2,0,3,2,2,2,2),4:(1,0,1,2,0,0,0),6:(1,0,1,2,2,0,0),5:(1,0,1,0,0,0,0),12:(1,1,4,2,2,4,0),7:(1,4,1,2,2,4,0)}
    expected.update({13:(1,1,4,2,2,4,0),14:(2,0,3,5,2,2,2),15:(3,3,1,2,2,2,0),16:(1,3,2,2,2,0,0),17:(1,0,1,2,2,2,0)})
    if s==1:expected[s]=(3,3,1 if not f and k==D and n in (W,F) else 2,2,2,0,0)
    if tuple(j['formats'][name] for name in ADDRESS_FIELDS)!=expected[s]:raise ValueError('Fixed SP3 tensor format mismatch')
    validate_job_ranges(j)
    return j
