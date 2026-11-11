"""Aligned typed view into a DMA allocation."""
import numpy as np

class View:
    def __init__(self,parent,offset,shape,dtype):
        self.root=parent.root;self.address=parent.address+offset
        self.shape=tuple(shape);self.dtype=np.dtype(dtype);self.nbytes=int(np.prod(shape))*self.dtype.itemsize
        if offset+self.nbytes>parent.nbytes or self.address%64:raise RuntimeError('Invalid buffer view')
        self.array=parent.array.view('u1').reshape(-1)[offset:offset+self.nbytes].view(dtype).reshape(shape)

