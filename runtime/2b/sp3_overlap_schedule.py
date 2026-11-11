"""Fixed CFG scheduling metadata; no hardware access or tensor arithmetic.

Each board executes one compute lane, one ordered TP lane and one SP lane.
Only completed producer programs may publish a DDR tile to a native reader.
"""
from dataclasses import dataclass
from types import MappingProxyType
from tp4_memory import SlotPool

SCHEDULE = 'Y3_CFG_OPERATOR_OVERLAP_V1'
CONDITIONS = ('positive', 'negative')
SCRATCH_NAMES = ('quant', 'sa', 'q', 'k', 'v', 'attn', 'ffn', 'ffn_scales')
OPERATOR_NAMES = ('self_attn', 'cross_attn', 'mlp')


@dataclass(frozen=True)
class BranchContext:
    generation: int
    step: int
    layer: int
    condition: str
    state: object
    tensors: object
    workspace: object
    gates: tuple

    @classmethod
    def create(cls, generation, step, layer, condition, state, tensors, workspace, gates):
        if condition not in CONDITIONS or not 0 <= layer < 28 or not 0 <= step < 19:
            raise ValueError('Fixed CFG branch context')
        if len(gates) != 3:
            raise ValueError('One immutable gate buffer per branch/operator is required')
        return cls(generation, step, layer, condition, state,
                   MappingProxyType(dict(tensors)), workspace, tuple(gates))

    def key(self, operator, first_row=0):
        return (self.step, self.layer, operator, CONDITIONS.index(self.condition), first_row)


@dataclass(frozen=True)
class ProjectionCompletion:
    context: BranchContext
    operator: int
    futures: tuple

    def wait(self):
        # Futures were submitted in row order to one TP worker. Reading every
        # result preserves the first exception; no wait releases a DMA slot.
        for future in self.futures:
            future.result()


class BranchSlotPools:
    """Two physical partial/sum slots per CFG branch; four slots in total."""
    def __init__(self):
        self.pools = {condition: SlotPool(2) for condition in CONDITIONS}

    def acquire(self, context, operator, first_row):
        pool = self.pools[context.condition]
        token = pool.acquire(context.key(operator, first_row), timeout=120)
        offset = CONDITIONS.index(context.condition) * 2
        return offset + token[0], token

    def release(self, condition, token):
        self.pools[condition].release(token)

    def fail(self, error):
        for pool in self.pools.values():
            pool.fail(error)

    @property
    def busy(self):
        return any(pool.busy for pool in self.pools.values())
