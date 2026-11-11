"""FIVE_FIXED64_V1: two AXI256 beats, fixed opcode/shape plus seven addresses."""
from collections import OrderedDict
import struct
from tp4_profile import validate_job
from c209_job_ranges import validate_job_ranges,validate_descriptor_span
from c209_tensor_layout import ADDRESS_FIELDS
from five_fixed_jobs import SUPPORTED_OPS

FORMAT='FIVE_FIXED64_V1'
MAGIC=0xC5
RECORD_BYTES=64
MAX_JOBS=128
ROWS=(128,256,2048,4096)
RECORD=struct.Struct('<8Q')
PROGRAM_ABI=dict(format=FORMAT,record_bytes=64,max_jobs=128,header_magic=0xC5,
    header_magic_bits=[31,24],header_reserved_zero_mask='0xffffffff00fc0000',
    address_fields=list(ADDRESS_FIELDS),shape_encoding='FIXED_APPLE_2B00_TP2_SP12_V1',supported_ops=list(SUPPORTED_OPS))


def pack_job(job):
    j=validate_job(job);op=j['stage'];m=j['rows']//8 if op==4 else j['rows'];shape=group=negative=0
    if op==1:
        shape={(2048,80,0):0,(1024,2048,0):1,(2048,1024,8):2,
               (4096,2048,0):3,(2048,4096,8):4,(64,2048,0):5}[j['cols'],j['inner'],j['flags']&~64]
    elif op==2:shape={68:0,1024:1}[j['cols']]
    elif op==7:
        shape=int(j['flags']==160)
        negative=int(shape and j['k_count']==11)
    elif op==13:group=0 if j['k_start']==0 else 1
    if op==1 and shape in (3,4):raise ValueError('Fixed FFN requires fused operators15/16')
    header=op|(shape<<8)|(ROWS.index(m)<<12)|(group<<14)|(negative<<16)|(int(bool(j['flags']&64) and op in (1,15,16))<<17)|(MAGIC<<24)
    return RECORD.pack(header,*(j[name] for name in ADDRESS_FIELDS))


def unpack_record(record):
    """Decode the fixed wire fields for host audit, without reconstructing V2."""
    if len(record)!=RECORD_BYTES:raise ValueError('Expected one 64-byte fixed record')
    values=RECORD.unpack(record);header=values[0]
    if header>>32 or (header>>24)&255!=MAGIC or header&0x00fc0000:raise ValueError('Fixed record magic/reserved bits')
    op=header&255;shape=(header>>8)&15;group=(header>>14)&3;negative=(header>>16)&1
    if op not in SUPPORTED_OPS:raise ValueError('Unknown fixed opcode')
    if (op==1 and shape in (3,4)) or (op==2 and shape==2):raise ValueError('Retired fixed compatibility phase')
    if ((op==1 and shape>5) or (op==2 and shape>1) or (op==7 and shape>1)
        or (op not in (1,2,7) and shape) or (op!=13 and group) or group>1
        or (negative and not(op==7 and shape==1))):raise ValueError('Unknown fixed shape/context')
    packed=bool(header&(1<<17))
    if packed != (op in (1,15,16)):raise ValueError('IP16 requires packed I16 for every GEMM')
    rows=ROWS[(header>>12)&3]
    partial=op==16 or op==1 and shape in (2,4)
    if rows not in ((128,256) if partial else (2048,4096)):raise ValueError('Wrong row code for fixed operation')
    if op==13 and (group==1 and rows!=2048):raise ValueError('Wrong global KV group/row binding')
    return dict(op=op,shape=shape,rows=rows,group=group,negative=bool(negative),packed_b_i16=packed,
                addresses=dict(zip(ADDRESS_FIELDS,values[1:])))


class PreparedFixedProgram:
    def __init__(self,jobs,payload):
        self.payload=payload;self.count=len(jobs)
        self.proofs=tuple(validate_job_ranges(j) for j in jobs)
        self._bound_descriptor=None
    def bind_descriptor(self,address,capacity):
        if capacity<len(self.payload):raise ValueError('Fixed descriptor storage is too short')
        binding=(address,capacity)
        if binding!=self._bound_descriptor:
            validate_descriptor_span(self.proofs,address,self.count,RECORD_BYTES)
            self._bound_descriptor=binding


class FixedProgramCache:
    def __init__(self,limit=512):self.limit=limit;self.entries=OrderedDict()
    def get(self,jobs):
        jobs=[validate_job(j) for j in jobs]
        if not 1<=len(jobs)<=MAX_JOBS:raise ValueError('A fixed program needs 1..128 jobs')
        # Key the exact validated scalar metadata before serializing it. Fixed
        # tensor addresses and op/shape tags recur across sampler steps; hits
        # retain their range proofs and need neither struct packing nor a new
        # descriptor proof. Validation above still rejects altered inputs.
        key=tuple(tuple((name,tuple(sorted(value.items())) if isinstance(value,dict) else value)
                        for name,value in sorted(j.items())) for j in jobs)
        entry=self.entries.get(key)
        if entry is None:
            payload=b''.join(pack_job(j) for j in jobs)
            entry=PreparedFixedProgram(jobs,payload);self.entries[key]=entry
            if len(self.entries)>self.limit:self.entries.popitem(last=False)
        else:self.entries.move_to_end(key)
        return entry
