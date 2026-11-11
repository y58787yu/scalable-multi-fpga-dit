"""Bind V2 tags to actual Buffer/View metadata without reading tensor values."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
import sys
import numpy as np

from c209_program import allowed_job_formats, validate_job
from c209_job_ranges import validate_job_ranges
from c209_tensor_layout import (ADDRESS_FIELDS, UNUSED, F16, F32, I16, U32, I8,
                                KV_PACKED, PackedKVLayout, validate_formats)


def buffer_metadata(buffer):
    """Validate the dtype/shape/address metadata of one real allocation/view."""
    try:
        dtype = np.dtype(buffer.dtype)
        shape = tuple(buffer.shape)
        address, nbytes = buffer.address, buffer.nbytes
        if (not shape or any(type(n) is not int or n < 0 for n in shape)
                or type(address) is not int or not 0 <= address < 1 << 64
                or type(nbytes) is not int or nbytes != math.prod(shape)*dtype.itemsize
                or address+nbytes > 1 << 64
                or dtype.byteorder == '>' or (dtype.byteorder == '=' and sys.byteorder != 'little')):
            raise ValueError('invalid scalar buffer metadata')
        array = getattr(buffer, 'array', None)
        if array is not None and (np.dtype(array.dtype) != dtype or tuple(array.shape) != shape
                                  or array.nbytes != nbytes or not array.flags.c_contiguous):
            raise ValueError('buffer metadata differs from its actual contiguous array')
        return dtype, address, nbytes
    except (AttributeError, TypeError, OverflowError) as error:
        raise ValueError('missing/invalid Buffer or View metadata') from error


def scalar_buffer_format(buffer):
    dtype, _, _ = buffer_metadata(buffer)
    formats = {('f',2):F16,('f',4):F32,('i',2):I16,('i',1):I8,('u',4):U32}
    try:
        return formats[dtype.kind, dtype.itemsize]
    except KeyError as error:
        raise ValueError('buffer dtype has no scalar V2 tag; raw bytes require an explicit composite binding') from error


@dataclass(frozen=True)
class PackedKVBinding:
    buffer: object
    layout: PackedKVLayout
    component: str

    def validate(self):
        if not isinstance(self.layout, PackedKVLayout) or self.component not in ('k','v'):
            raise ValueError('packed KV requires an explicit K/V layout component')
        dtype, address, nbytes = buffer_metadata(self.buffer)
        if dtype != np.dtype('u1') or nbytes < self.layout.sizes[self.component]:
            raise ValueError('packed KV byte allocation does not cover its declared layout')
        return address, nbytes


def bind_job(job, buffers):
    """Return a fully tagged job using explicit slot -> Buffer/View bindings.

Ordinary F16/F32 is taken from the actual dtype. Semantic constraints come
from stage/flags/mode, never from a tensor name or a guessed byte address.
    Conservative execution ranges (including complete AXI beats) must fit the
    bound allocations. Unsafe read/write aliasing is rejected before dispatch.
    This does not replace operation-specific shape validation.
"""
    if not isinstance(job, Mapping) or not isinstance(buffers, Mapping):
        raise ValueError('job and explicit buffer bindings must be mappings')
    rules = allowed_job_formats(job.get('stage',0),job.get('flags',0),job.get('mode',0))
    active = {name for name,formats in rules.items() if formats != (0,)}
    if set(buffers) != active:
        raise ValueError('buffer bindings must cover exactly the active semantic slots')
    if job.get('stage') in (1,15,16) and job.get('flags',0)&64:
        weight=buffers.get('b')
        if (getattr(weight,'packed_b_layout',None)!='C209_B_N640_K128_SHARD128_I16_V1'
            or getattr(weight,'logical_shape',None)!=(job.get('cols'),job.get('inner'))):
            raise ValueError('Packed GEMM requires an explicitly bound matching I16 panel asset')
    result = dict(job)
    tags = {name:int(UNUSED) for name in ADDRESS_FIELDS}
    for name,binding in buffers.items():
        if isinstance(binding,PackedKVBinding):
            address,nbytes = binding.validate()
            expected = 'k' if name in ('b','c') else ('v' if name=='aux2' else None)
            if rules[name] != (int(KV_PACKED),) or binding.component != expected:
                raise ValueError('packed KV binding component differs from stage slot')
            tag = KV_PACKED
            if result.get(name,address) != address:
                raise ValueError('packed KV base cannot use an implicit scalar offset')
        else:
            _,address,nbytes = buffer_metadata(binding)
            tag = scalar_buffer_format(binding)
        pointer = result.setdefault(name,address)
        if type(pointer) is not int or not address <= pointer < address+nbytes:
            raise ValueError('descriptor address is outside its explicitly bound buffer: '+name)
        tags[name] = int(tag)
    if 'formats' in result and validate_formats(result['formats']) != tags:
        raise ValueError('explicit descriptor tags disagree with bound buffer dtypes')
    result['formats'] = tags
    result=validate_job(result)
    proof=validate_job_ranges(result)
    for span in proof['ranges']:
        binding=buffers[span['slot']]
        buffer=binding.buffer if isinstance(binding,PackedKVBinding) else binding
        _,base,nbytes=buffer_metadata(buffer)
        if span['start']<base or span['end']>base+nbytes:
            raise ValueError('complete AXI access exceeds bound allocation: '+span['slot'])
    return result
