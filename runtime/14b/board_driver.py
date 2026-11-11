"""ARM/C ABI structures and guarded DMA allocations."""
import ctypes as C
import numpy as np

class RegWrite(C.Structure):_fields_=[('offset',C.c_uint32),('value',C.c_uint32)]


class CoreJob(C.Structure):_fields_=[('kernel',C.c_uint64),('settings',C.POINTER(RegWrite)),('count',C.c_uint32),('error_reg',C.c_uint32),('tiles_reg',C.c_uint32),('timeout_ms',C.c_uint32)]


class ExchangeJob(C.Structure):_fields_=[('dma',C.c_uint64),('link',C.c_uint64),('tx_phys',C.c_uint64),('rx_phys',C.c_uint64),('bytes',C.c_uint32),('count',C.c_uint32),('transaction',C.c_uint32),('meta',C.c_uint32),('timeout_ms',C.c_uint32)]


class Stats(C.Structure):_fields_=[('elapsed_ns',C.c_uint64),('error',C.c_uint32),('tiles',C.c_uint32),('rx_bytes',C.c_uint32),('rx_errors',C.c_uint32),('tx_frames',C.c_uint32),('rx_frames',C.c_uint32)]


class Buffer:
    def __init__(self,allocate,shape,dtype,data=None):
        self.shape=tuple(shape);self.dtype=np.dtype(dtype);self.nbytes=int(np.prod(shape))*self.dtype.itemsize
        self.root=allocate(shape=(self.nbytes+256,),dtype='u1');self.root[:128]=0xa5;self.root[-128:]=0x5a
        self.array=self.root[128:128+self.nbytes].view(self.dtype).reshape(shape)
        self.address=int(self.root.physical_address)+128
        if self.address%64:raise RuntimeError('Unaligned allocation')
        self.array[:]=0 if data is None else data;self.root.flush()
    def check(self):
        self.root.invalidate()
        if not np.all(self.root[:128]==0xa5) or not np.all(self.root[-128:]==0x5a):raise RuntimeError('Allocation guard corrupted')
    def close(self):self.root.freebuffer()


class Board:
    def make(self,shape,dtype,data=None):
        b=Buffer(self.allocate,shape,dtype,data);self.buffers.append(b);return b
