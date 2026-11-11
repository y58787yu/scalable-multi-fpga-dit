"""Exact SP3 row and packed-KV byte layout; no hardware imports."""
from dataclasses import dataclass
from c209_tensor_layout import PackedKVLayout

ROWS=(1408,1408,1280)
STARTS=(0,1408,2816)
CONDITIONS=('positive','negative')

@dataclass(frozen=True)
class SequenceRole:
    group:int
    rank:int
    def __post_init__(self):
        if type(self.group)is not int or self.group not in range(3) or type(self.rank)is not int or self.rank not in range(4):raise ValueError('SP3 role')
    @property
    def rows(self):return ROWS[self.group]
    @property
    def start(self):return STARTS[self.group]
    @property
    def token_offset(self):return self.start
    @property
    def node(self):return self.rank+1
    @property
    def stage(self):return self.group
    @property
    def owned_layers(self):return range(self.group*12,(self.group+1)*12)

def tiles(rows):
    if rows not in ROWS:raise ValueError('Exact SP3 rows required')
    return tuple((start,min(256,rows-start)) for start in range(0,rows,256))

def kv_spans(group,heads=10):
    """(part,local byte offset,global byte offset,bytes), including each V head.

    K is head/token-major. V and V scale are head-major, then global128-token
    blocks. Each head's local slice scatters into the final global allocation.
    """
    if group not in range(3):raise ValueError('SP group')
    local=PackedKVLayout(ROWS[group],heads);full=PackedKVLayout(4096,heads)
    for head in range(heads):
        yield 'k',head*ROWS[group]*256,(head*4096+STARTS[group])*256,ROWS[group]*256
    yield 'ks',0,STARTS[group]*full.scale_stride*4,local.sizes['ks']
    for part,block_bytes in (('v',32768),('vs',512)):
        size=local.blocks*block_bytes
        for head in range(heads):
            yield part,head*size,(head*full.blocks+STARTS[group]//128)*block_bytes,size

def scatter_local(local,full,group,heads=10):
    """Copy already quantized bytes only: no requantization or arithmetic."""
    for part,src,dst,n in kv_spans(group,heads):
        a=memoryview(local[part]).cast('B');b=memoryview(full[part]).cast('B')
        if src+n>len(a) or dst+n>len(b):raise ValueError('Packed KV capacity')
        b[dst:dst+n]=a[src:src+n]

def layer_schedule():
    return tuple((layer,condition) for layer in range(36) for condition in CONDITIONS)
