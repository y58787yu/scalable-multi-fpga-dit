"""MODULAR16 byte-layout and scalar-codec reference; no board/native imports.

Addresses are bytes, row strides are scalar elements. F16 conversion starts
from FP32 and implements the finite-value RNE/FTZ contract in
source/include/c209_precision.hpp, including its minimum-normal midpoint.
This is a software specification, not evidence that native RTL implements it.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
import math
import struct


class TensorFormat(IntEnum):
    UNUSED = 0
    F16 = 1
    F32 = 2
    I16 = 3
    KV_PACKED = 4
    U32 = 5
    I8 = 6


UNUSED, F16, F32, I16, KV_PACKED, U32, I8 = TensorFormat
ADDRESS_FIELDS = ('a', 'b', 'c', 'aux0', 'aux1', 'aux2', 'aux3')
ROUNDING = 'RNE_FTZ_MIN_NORMAL_MIDPOINT_TO_ZERO_V1'
ROLE_FORMATS = {'ordinary': F16, 'partial': F32, 'state': F32,
                'scale': F32, 'metadata': F32, 'cpu_sampler': F32,
                'matrix_operand': I16, 'prepared_kv': KV_PACKED,
                'bitmap': U32, 'packed_weight_i8': I8}


def _integer(value, name, maximum=(1 << 64) - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(name + ' must be a bounded nonnegative integer')
    return value


def tensor_format(value):
    if type(value) not in (int, TensorFormat):
        raise ValueError('tensor format must be an explicit enum/integer, not a dtype guess')
    try:
        return TensorFormat(value)
    except ValueError as error:
        raise ValueError('unknown tensor format: ' + str(value)) from error


def validate_formats(formats):
    """Return all seven numeric format tags; omissions are never inferred."""
    if not isinstance(formats, Mapping) or set(formats) != set(ADDRESS_FIELDS):
        raise ValueError('formats must explicitly contain exactly ' + ', '.join(ADDRESS_FIELDS))
    return {name: int(tensor_format(formats[name])) for name in ADDRESS_FIELDS}


def format_for_role(role):
    try:
        return ROLE_FORMATS[role]
    except (KeyError, TypeError) as error:
        raise ValueError('unknown explicit tensor role') from error


def validate_role(value, role):
    actual = tensor_format(value)
    if actual != format_for_role(role):
        raise ValueError('tensor format does not match explicit role: ' + role)
    return actual


def f32_bits(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not a tensor scalar')
    try:
        return struct.unpack('<I', struct.pack('<f', float(value)))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise ValueError('value cannot be represented as FP32') from error


def f16_bits(value):
    """FP32 -> binary16 RNE, with the precise C209 finite-value FTZ rule.

Infinity retains its sign. NaNs become quiet canonical NaNs; native NaN
payload equivalence is deliberately not claimed.
"""
    bits = f32_bits(value)
    sign, magnitude = (bits >> 16) & 0x8000, bits & 0x7fffffff
    if magnitude <= 0x387fe000:
        return sign
    if magnitude < 0x38800000:
        return sign | 0x0400
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    if exponent == 255:
        return sign | (0x7e00 if fraction else 0x7c00)
    half_exponent = exponent - 127 + 15
    if half_exponent >= 31:
        return sign | 0x7c00
    body = (half_exponent << 10) | (fraction >> 13)
    remainder = fraction & 0x1fff
    if remainder > 0x1000 or (remainder == 0x1000 and body & 1):
        body += 1
    return sign | body


def f16_from_bits(bits):
    """Binary16 -> Python float, flushing subnormal encodings on read too."""
    _integer(bits, 'binary16 bits', 65535)
    sign, exponent, fraction = (-1.0 if bits & 0x8000 else 1.0), (bits >> 10) & 31, bits & 1023
    if exponent == 0:
        return math.copysign(0.0, sign)
    if exponent == 31:
        return math.copysign(math.nan if fraction else math.inf, sign)
    return sign * math.ldexp(1024 + fraction, exponent - 25)


def round16(value):
    return f16_from_bits(f16_bits(value))


def element_bytes(value):
    value = tensor_format(value)
    if value not in (F16, F32, I16, U32, I8):
        raise ValueError('UNUSED/KV_PACKED has no ordinary scalar element size')
    return 1 if value == I8 else (4 if value in (F32, U32) else 2)


def encode_scalar(value, format):
    format = tensor_format(format)
    if format == F16:
        return struct.pack('<H', f16_bits(value))
    if format == F32:
        return struct.pack('<I', f32_bits(value))
    if format == I16:
        if type(value) is not int or not -32768 <= value <= 32767:
            raise ValueError('I16 scalar must be an exact signed 16-bit integer')
        return struct.pack('<h', value)
    if format == I8:
        if type(value) is not int or not -128 <= value <= 127:
            raise ValueError('I8 scalar must be an exact signed 8-bit integer')
        return struct.pack('<b', value)
    if format == U32:
        return struct.pack('<I', _integer(value, 'U32 scalar', (1 << 32) - 1))
    raise ValueError('scalar codec requires an explicit scalar format')


def decode_scalar(payload, format):
    format = tensor_format(format)
    if len(payload) != element_bytes(format):
        raise ValueError('scalar byte length differs from explicit format')
    if format == F16:
        return f16_from_bits(struct.unpack('<H', payload)[0])
    return struct.unpack({F32: '<f', I16: '<h', I8: '<b', U32: '<I'}[format], payload)[0]


@dataclass(frozen=True)
class TensorLayout:
    """A scalar tensor view. Flat values use logical row-major ordering.

Only logical columns are touched by pack_into; row padding, tail lanes and
allocation canaries remain unchanged. Buffer origins are explicit byte
addresses, allowing one arena to model real descriptor addresses.
"""
    format: TensorFormat
    rows: int
    cols: int
    stride: int
    address: int = 0

    def __post_init__(self):
        object.__setattr__(self, 'format', tensor_format(self.format))
        size = element_bytes(self.format)
        for name in ('rows', 'cols', 'stride', 'address'):
            _integer(getattr(self, name), name)
        if self.stride < self.cols or self.address % size:
            raise ValueError('invalid element stride or scalar byte alignment')
        if self.address + self.allocation_bytes > 1 << 64:
            raise ValueError('tensor address range exceeds uint64')

    @property
    def element_bytes(self):
        return element_bytes(self.format)

    @property
    def allocation_bytes(self):
        return self.rows * self.stride * self.element_bytes

    @property
    def span_bytes(self):
        if not self.rows or not self.cols:
            return 0
        return ((self.rows - 1) * self.stride + self.cols) * self.element_bytes

    def byte_address(self, row, column):
        _integer(row, 'row'); _integer(column, 'column')
        if row >= self.rows or column >= self.cols:
            raise ValueError('tensor index is outside logical shape')
        return self.address + (row * self.stride + column) * self.element_bytes

    def _view(self, buffer, buffer_address, *, writable=False):
        _integer(buffer_address, 'buffer byte address')
        try:
            view = memoryview(buffer).cast('B')
        except (TypeError, ValueError) as error:
            raise ValueError('tensor buffer must be contiguous bytes') from error
        start = self.address - buffer_address
        if start < 0 or start + self.span_bytes > len(view) or (writable and view.readonly):
            raise ValueError('tensor view exceeds buffer or buffer is not writable')
        return view, start

    def pack_into(self, buffer, values, *, buffer_address=0):
        view, start = self._view(buffer, buffer_address, writable=True)
        values = list(values)
        if len(values) != self.rows * self.cols:
            raise ValueError('logical value count differs from tensor shape')
        # Validate and snapshot every scalar before the first write. This is
        # also safe when the caller's input generator reads the same arena.
        encoded = [encode_scalar(value, self.format) for value in values]
        for index, payload in enumerate(encoded):
            row, column = divmod(index, self.cols)
            offset = start + (row * self.stride + column) * self.element_bytes
            view[offset:offset + self.element_bytes] = payload
        return buffer

    def unpack_from(self, buffer, *, buffer_address=0):
        view, start = self._view(buffer, buffer_address)
        return [decode_scalar(view[start + (r*self.stride+c)*self.element_bytes:
                                   start + (r*self.stride+c+1)*self.element_bytes], self.format)
                for r in range(self.rows) for c in range(self.cols)]

    def pack(self, values):
        output = bytearray(self.allocation_bytes)
        self.pack_into(output, values, buffer_address=self.address)
        return bytes(output)


def copy_tensor(source, source_layout, destination, destination_layout,
                *, source_address=0, destination_address=0):
    """Explicit format conversion with snapshot semantics, including overlap."""
    if (source_layout.rows, source_layout.cols) != (destination_layout.rows, destination_layout.cols):
        raise ValueError('tensor copy logical shapes differ')
    values = source_layout.unpack_from(source, buffer_address=source_address)
    return destination_layout.pack_into(destination, values, buffer_address=destination_address)


@dataclass(frozen=True)
class PackedKVLayout:
    """Four-buffer head/token K and head/block/column V layout.

KV_PACKED is a composite contract, not an alternate spelling for a flat
two-byte tensor. Individual K/V scalars are I16 and scale scalars are F32.
"""
    tokens: int
    heads: int = 8

    def __post_init__(self):
        _integer(self.tokens, 'stored tokens', 4096)
        _integer(self.heads, 'heads', 16)
        if not self.heads:
            raise ValueError('prepared KV requires at least one head')

    @property
    def blocks(self):
        return (self.tokens + 127) // 128

    @property
    def scale_stride(self):
        return (self.heads + 7) // 8 * 8

    @property
    def sizes(self):
        return dict(k=max(32, self.tokens*self.heads*128*2),
                    v=max(32, self.heads*self.blocks*128*128*2),
                    ks=max(32, self.tokens*self.scale_stride*4),
                    vs=max(32, self.heads*self.blocks*128*4))

    @property
    def guarded_bytes(self):
        return sum((size+256+4095)//4096*4096 for size in self.sizes.values())

    def byte_offset(self, component, token, head, column=0):
        for name, value in (('token', token), ('head', head), ('column', column)):
            _integer(value, name)
        if token >= self.tokens or head >= self.heads or column >= 128:
            raise ValueError('prepared KV index is outside logical shape')
        if component == 'k':
            # SAMEHEAD_K_HEAD_TOKEN_V1: compact cross11/16 retains capacity;
            # self4096 supplies contiguous per-head128-key DDR bursts.
            return ((head*self.tokens+token)*128+column)*2
        if component == 'v':
            return (((head*self.blocks+token//128)*128+column)*128+token%128)*2
        if component == 'ks':
            if column:
                raise ValueError('K scale is one scalar per token/head')
            return (token*self.scale_stride+head)*4
        if component == 'vs':
            return ((head*self.blocks+token//128)*128+column)*4
        raise ValueError('unknown prepared KV component')
