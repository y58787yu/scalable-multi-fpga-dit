"""IP16 integrated fixed-apple jobs and explicit numerical policies.

Typed host jobs are encoded as FIVE_FIXED64_V1 for the new image, not V2 wire
descriptors or emulation on the old SP3 bitstream.
All arithmetic and quantization stay in PL; host code binds real buffer metadata.
"""
from functools import lru_cache
from types import MappingProxyType
from c209_typed_job import bind_job,PackedKVBinding

PREPARE_KV_GLOBAL=13
NORM_ADALN_FIXED=14
GEMM_GELU_F16=15
GEMM_F16_A_PARTIAL=16
RMS_ROPE_FUSED=17
SUPPORTED_OPS=[1,2,4,7,13,14,15,16,17]
FIXED_OPS=dict(version=1,stages=[13,14,15,16,17],shared_prefix=True,kv_global=True,
    ffn_gelu_fused=True,f16_a_quantized_on_read=True,rms_rope_fused=True,modulation_gain_precomputed=True,
    adaln_scale_policy='SCL1_ANALYTIC_POW2_V1',kv_value_scale_policy='PV_PROB_SCALE_FOLDED_V1',
    norm_reduction='FP32_BANK_LOCAL16_V1')
LEGACY_ASSET_PROFILE=None


def require_image_contract(protocol):
    from five_program import PROGRAM_ABI
    if (protocol.get('hardware_id')!='0xC2092B10' or protocol.get('runtime_id')!='0xC2092B11'
            or protocol.get('compute_hardware_id')!='0xC2092B00'
            or protocol.get('compute_ip_version',protocol.get('ip_version'))!='20.0'
            or protocol.get('fixed_apple_ops')!=FIXED_OPS
            or protocol.get('descriptor_format')!='FIVE_FIXED64_V1'
            or protocol.get('program_abi')!=PROGRAM_ABI):
        raise ValueError('TP2 dual-port runtime requires the IP20/2B00 compute and 2B10 shell contract')
    return True


@lru_cache(maxsize=32)
def scalar_template(stage,rows,token_offset=0):
    if rows not in (128,256,2048,4096):raise ValueError('Unknown fixed row shape')
    base=dict(stage=stage,rows=rows,flags=0,mode=0,factor=0.,epsilon=1e-6)
    if stage==PREPARE_KV_GLOBAL:
        if (token_offset,rows) not in ((0,4096),(0,2048),(2048,2048)):raise ValueError('Wrong SP group slice')
        base.update(cols=4096,heads=8,k_start=token_offset,k_count=rows,lda=1024,ldb=1024,ldc=1024)
    elif stage==NORM_ADALN_FIXED:base.update(cols=2048,lda=2048,ldc=2048)
    elif stage==GEMM_GELU_F16:base.update(cols=4096,inner=2048,lda=2048,ldb=2048,ldc=4096,flags=64)
    elif stage==GEMM_F16_A_PARTIAL:
        if rows not in (128,256):raise ValueError('FFN down requires a partial output tile')
        base.update(cols=2048,inner=4096,lda=4096,ldb=4096,ldc=2048,flags=72)
    elif stage==RMS_ROPE_FUSED:base.update(cols=1024,lda=1024,ldc=1024,heads=8)
    else:raise ValueError('Unknown fixed operator')
    if stage!=GEMM_F16_A_PARTIAL and rows not in (2048,4096):raise ValueError('Expected the full local SP slice')
    return MappingProxyType(base)


class FixedJobs:
    def __init__(self,role):self.rows,self.token_offset=role.rows,role.token_offset
    def norm(self,state,quant,shift,one_plus_modulation,scales,*,scale_policy):
        return bind_job(dict(scalar_template(NORM_ADALN_FIXED,self.rows)),
                        dict(a=state,c=quant,aux0=scale_policy,aux1=shift,aux2=one_plus_modulation,aux3=scales))
    def prepare_kv(self,k,v,workspace):
        b=workspace.buffers;l=workspace.layout
        if l.tokens!=4096 or l.heads!=8:raise ValueError('Direct KV destination must have global 4096x10 layout')
        return bind_job(dict(scalar_template(PREPARE_KV_GLOBAL,self.rows,self.token_offset)),
            dict(a=k,b=v,c=PackedKVBinding(b['k'],l,'k'),aux0=b['ks'],aux1=b['vs'],aux2=PackedKVBinding(b['v'],l,'v')))
    def rms_rope(self,value,gamma,cos,sin):
        return bind_job(dict(scalar_template(RMS_ROPE_FUSED,self.rows)),dict(a=value,c=value,aux0=gamma,aux1=cos,aux2=sin))
    def ffn_up(self,quant,weight,output,input_scales,output_scales):
        return bind_job(dict(scalar_template(GEMM_GELU_F16,self.rows)),
            dict(a=quant,b=weight[0],c=output,aux0=input_scales,aux1=weight[1],aux2=output_scales))
    def ffn_down(self,source,weight,output,scales,first,rows):
        job=dict(scalar_template(GEMM_F16_A_PARTIAL,rows),a=source.address+first*4096*2,aux0=scales.address+first*4)
        return bind_job(job,dict(a=source,b=weight[0],c=output,aux0=scales,aux1=weight[1]))
