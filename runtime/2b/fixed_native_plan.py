"""Prevalidated fixed layer commands and bounded native projection ABI.

Plans retain Buffer owners through their model. Installing a plan only changes
the finite 16 KiB descriptor allocation; no live plan may be overwritten.
"""
import ctypes as C
from types import MappingProxyType
from five_program import FixedProgramCache, RECORD_BYTES, MAX_JOBS
from board_driver import CoreJob, Stats


PLAN_BYTES=16384

COUNTER_NAMES=('elapsed','multiply_issue','load_a_pending','load_b_pending_after_a',
               'seed','store','output_blocked','other')


class CoreCompletion(C.Structure):
    _fields_=[('stats',Stats),('cycles',C.c_uint32*8),
              ('start_ns',C.c_uint64),('end_ns',C.c_uint64)]


class PreparedCommand:
    def __init__(self,jobs,prepared,offset):
        self.jobs=tuple(MappingProxyType(dict(j,formats=MappingProxyType(dict(j['formats'])))) for j in jobs)
        self.prepared=prepared;self.offset=offset;self.plan=None;self.native=None
    def __iter__(self):return iter(self.jobs)
    def __len__(self):return len(self.jobs)


class LayerProgramPlan:
    def __init__(self,key):
        self.key=key;self.commands={};self._payload=bytearray();self._unique={};self.sealed=False
        self._cache=FixedProgramCache(limit=MAX_JOBS)
    def add(self,key,jobs):
        if self.sealed or key in self.commands:raise RuntimeError('Fixed plan is immutable and keys are unique')
        jobs=tuple(jobs);prepared=self._cache.get(jobs)
        command=self._unique.get(prepared.payload)
        if command is None:
            command=PreparedCommand(jobs,prepared,len(self._payload));command.plan=self
            self._payload.extend(prepared.payload);self._unique[prepared.payload]=command
        self.commands[key]=command
        return command
    def seal(self):
        if not self._payload or len(self._payload)>PLAN_BYTES:
            raise ValueError('Fixed layer exceeds finite 16 KiB descriptor allocation')
        self.payload=bytes(self._payload);self.commands=MappingProxyType(self.commands)
        self.sealed=True;self._cache=None;self._payload=None
        return self
    def __getitem__(self,key):return self.commands[key]
    def bind(self,address):
        if not self.sealed:raise RuntimeError('Unsealed execution plan')
        for command in self._unique.values():
            command.prepared.bind_descriptor(address+command.offset,len(command.prepared.payload))


def projection_type(pair_job,stream_stats):
    class NativeProjection(C.Structure):
        _fields_=[('controller',C.c_uint64),('cores',C.POINTER(CoreJob)),
                  ('round0',C.POINTER(pair_job)),('round1',C.POINTER(pair_job)),
                  ('bases',C.POINTER(C.c_uint32)),('counter_regs',C.POINTER(C.c_uint32)),
                  ('compute',C.POINTER(CoreCompletion)),('reduce',C.POINTER(stream_stats)),
                  ('tp_start_ns',C.POINTER(C.c_uint64)),('tp_end_ns',C.POINTER(C.c_uint64)),
                  ('count',C.c_uint32),('timeout_ms',C.c_uint32),
                  ('slots',C.c_uint32*2),('failed',C.c_int32),
                  ('produced',C.c_uint32),('retired',C.c_uint32)]
    return NativeProjection
