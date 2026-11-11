"""C209 MODULAR16 V2 program descriptors; pure serialization, no hardware.

Each 128-byte little-endian record is four AXI256 words. The first 112 bytes
retain their field offsets. Bytes 112..123 carry mandatory version/format
tags, and bytes 124..127 are reserved zero. Addresses are bytes and strides
are elements. Explicit F32 ordinary tensors support migration A/B tests;
the new runtime's ordinary storage profile is F16, not inferred from names.
"""
from collections.abc import Mapping
import math
import struct

from c209_tensor_layout import (ADDRESS_FIELDS, TensorFormat, UNUSED, F16, F32,
                                I16, KV_PACKED, U32, I8, validate_formats)

EXECUTE_PROGRAM = 11
RECORD_BYTES = 128
MAX_JOBS = 128
DESCRIPTOR_MAGIC = 0x324D3943  # little-endian bytes: C9M2
DESCRIPTOR_VERSION = 2
_U32 = ('stage', 'rows', 'cols', 'inner', 'lda', 'ldb', 'ldc', 'heads',
        'flags', 'mode', 'k_start', 'k_count')
_U64 = ADDRESS_FIELDS
_DEFAULTS = dict(stage=0, rows=1, cols=8, inner=0, lda=8, ldb=8, ldc=8,
                 heads=0, flags=0, mode=0, k_start=0, k_count=0,
                 a=0, b=0, c=0, aux0=0, aux1=0, aux2=0, aux3=0,
                 factor=0., epsilon=1e-6)
_RECORD = struct.Struct('<12I7Q2fIB7B4x')
assert _RECORD.size == RECORD_BYTES
_FLAGS = {1: 1|2|8|16|64, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0,
          7: 128|32, 8: 0, 9: 0, 10: 0, 12: 32,13:0,14:0,15:64,16:8|64,17:0}
PROGRAM_ABI = dict(version=DESCRIPTOR_VERSION, record_bytes=RECORD_BYTES,
                   magic=DESCRIPTOR_MAGIC, magic_offset=112, version_offset=116,
                   format_offset=117, format_fields=list(ADDRESS_FIELDS),
                   reserved_zero_offset=124, reserved_zero_bytes=4,
                   addresses='bytes', strides='elements',
                   format_tags={value.name: int(value) for value in TensorFormat})


def _uint(value, bits, name):
    if type(value) is not int:
        raise TypeError(name + ' must be an integer')
    if not 0 <= value < (1 << bits):
        raise ValueError(name + ' is outside uint' + str(bits))
    return value


def allowed_job_formats(stage, flags=0, mode=0):
    """Allowed tags by semantic slot; this never fills missing job formats.

Metadata/scales are F32; only explicitly ordinary float slots admit F16 or
F32. LOCAL_PARTIAL c is always F32. KV_PACKED has its composite addressing.
"""
    _uint(stage, 32, 'stage'); _uint(flags, 32, 'flags'); _uint(mode, 32, 'mode')
    if stage not in _FLAGS:
        raise ValueError('program jobs must use admitted local stages; nesting is forbidden')
    if flags & ~_FLAGS[stage]:
        raise ValueError('unsupported flag bits for program stage')
    if (stage == 5 and mode > 7) or (stage == 10 and mode > 3) or (stage not in (5, 10) and mode):
        raise ValueError('unsupported mode for program stage')
    rules = {name: (int(UNUSED),) for name in ADDRESS_FIELDS}

    def put(name, *formats):
        rules[name] = tuple(int(value) for value in formats)

    def ordinary(*names):
        for name in names:
            put(name, F16, F32)

    if stage == 1:
        if flags & 8 and flags & (1|2):
            raise ValueError('LOCAL_PARTIAL cannot apply bias or GELU')
        put('a', I16); put('b', I16)  # IP16 bit64 selects packed I16, never I8
        put('c', F32) if flags & 8 else ordinary('c')
        put('aux0', F32); put('aux1', F32)
        if flags & 1: put('aux2', F32)
        if flags & 16: put('aux3', U32)
    elif stage in (2, 8):
        ordinary('a'); put('c', I16); put('aux0', F32)
    elif stage in (3, 4):
        ordinary('a', 'c'); put('aux0', F32)
        if flags & 1: put('aux1', F32)
    elif stage == 5:
        ordinary('a', 'c')
        if mode in (0, 1, 3, 7): ordinary('b')
        if mode in (2, 3, 7): put('aux0', F32)
        if mode == 2: put('aux1', F32)
        if mode == 7: ordinary('aux1')
    elif stage == 6:
        ordinary('a', 'c'); put('aux0', F32); put('aux1', F32)
    elif stage == 7:
        if not flags & 128:
            raise ValueError('attention requires PREQUANT_KV')
        ordinary('a', 'c'); put('b', KV_PACKED)
        put('aux0', F32); put('aux1', F32); put('aux2', KV_PACKED)
    elif stage == 9:
        put('a', I16); put('c', F32); put('aux0', F32)
    elif stage == 10:
        ordinary('a')
        for name in ('aux0', 'aux1', 'aux2'): put(name, F32)
        if mode & 2:
            ordinary('c')
        else:
            put('c', I16); put('aux3', F32)
    elif stage == 12:
        ordinary('a', 'b'); put('c', KV_PACKED)
        put('aux0', F32); put('aux1', F32); put('aux2', KV_PACKED)
    elif stage == 13:
        put('a',F16);put('b',F16);put('c',KV_PACKED)
        put('aux0',F32);put('aux1',F32);put('aux2',KV_PACKED)
    elif stage == 14:
        put('a',F32);put('c',I16);put('aux0',U32);put('aux1',F32);put('aux2',F32);put('aux3',F32)
    elif stage == 15:
        put('a',I16);put('b',I16);put('c',F16);put('aux0',F32);put('aux1',F32);put('aux2',F32)
    elif stage == 16:
        put('a',F16);put('b',I16);put('c',F32);put('aux0',F32);put('aux1',F32)
    elif stage == 17:
        put('a',F16);put('c',F16);put('aux0',F32);put('aux1',F32);put('aux2',F32)
    return rules


def validate_job(job: Mapping) -> dict:
    """Validate every encoded field and return the normalized V2 record.

Operation-specific dimensions and buffer capacities also need native/host
range checks; serialization alone does not prove a job's numerical validity.
"""
    if not isinstance(job, Mapping):
        raise TypeError('job must be a mapping')
    extra = set(job) - (set(_DEFAULTS) | {'formats'})
    if extra:
        raise ValueError('unsupported program fields: ' + repr(sorted(extra, key=str)))
    if 'formats' not in job:
        raise ValueError('V2 jobs require explicit formats; legacy defaults are forbidden')
    fields = dict(_DEFAULTS, **job)
    fields['formats'] = validate_formats(fields['formats'])
    for name in _U32:
        _uint(fields[name], 32, name)
    for name in _U64:
        _uint(fields[name], 64, name)
        if fields[name] & 31:
            raise ValueError('all descriptor base addresses must be 32-byte aligned')
    for name in ('factor', 'epsilon'):
        value = fields[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(name + ' must be a finite FP32 number')
        try:
            fields[name] = struct.unpack('<f', struct.pack('<f', value))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(name + ' cannot be represented in FP32') from error
    rules = allowed_job_formats(fields['stage'], fields['flags'], fields['mode'])
    for name in ADDRESS_FIELDS:
        actual = fields['formats'][name]
        if actual not in rules[name]:
            raise ValueError('format does not match stage/flags/mode role: ' + name)
        if actual == UNUSED and fields[name] != 0:
            raise ValueError('UNUSED address must be zero: ' + name)
    return fields


def pack_job(job: Mapping) -> bytes:
    fields = validate_job(job)
    return _RECORD.pack(*(fields[name] for name in _U32),
                        *(fields[name] for name in _U64), fields['factor'], fields['epsilon'],
                        DESCRIPTOR_MAGIC, DESCRIPTOR_VERSION,
                        *(fields['formats'][name] for name in ADDRESS_FIELDS))


def unpack_job(record: bytes) -> dict:
    if len(record) != RECORD_BYTES:
        raise ValueError('one program record must contain exactly 128 bytes')
    if any(record[124:]):
        raise ValueError('V2 program record reserved bytes must be zero')
    values = _RECORD.unpack(record)
    if values[21] != DESCRIPTOR_MAGIC or values[22] != DESCRIPTOR_VERSION:
        raise ValueError('missing/unknown V2 descriptor magic or version')
    result = dict(zip((*_U32, *_U64, 'factor', 'epsilon'), values[:21]))
    result['formats'] = dict(zip(ADDRESS_FIELDS, values[23:30]))
    return validate_job(result)


def pack_program(jobs) -> bytes:
    jobs = list(jobs)
    if not 1 <= len(jobs) <= MAX_JOBS:
        raise ValueError('a program must contain 1..128 jobs')
    return b''.join(pack_job(job) for job in jobs)


def unpack_program(payload: bytes) -> list:
    if not payload or len(payload) % RECORD_BYTES or len(payload) > MAX_JOBS * RECORD_BYTES:
        raise ValueError('program payload must contain 1..128 complete records')
    return [unpack_job(payload[offset:offset + RECORD_BYTES])
            for offset in range(0, len(payload), RECORD_BYTES)]
