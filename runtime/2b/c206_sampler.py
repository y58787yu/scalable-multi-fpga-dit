"""Portable FP32 sampling for complete Cosmos Karras/Euler schedules.

Tables and validation vectors are exported by Diffusers 0.34.0. No scheduler is
reconstructed from an approximate formula on a board. Every ufunc below writes
an FP32 result before the next operation; in particular CFG is not rewritten
and multiply/add operations are not fused.
"""
from dataclasses import dataclass
import hashlib
import math
import numpy as np


SEED = 20261006
GUIDANCE = 7
POINTS, UPDATES = 20, 19


def f32(value):
    return np.asarray(value, dtype='<f4')


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f4').tobytes()).hexdigest()


def require_finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite C206 '+name)


def patches(latent):
    value = f32(latent)
    if value.ndim != 5 or value.shape[:3] != (1, 16, 1) or value.shape[3] != value.shape[4] or value.shape[3] % 2:
        raise ValueError('Expected square Cosmos latent [1,16,1,H,W]')
    side = value.shape[-1]//2
    packed = value.reshape(1, 16, 1, 1, side, 2, side, 2).transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(side*side, 64)
    result = np.zeros((side*side, 80), dtype='<f4')
    result[:, :64] = packed
    return result


def unpatch(prediction):
    value = f32(prediction)
    if value.ndim != 2 or value.shape[1] != 64:
        raise ValueError('Expected full Cosmos prediction [tokens,64]')
    side = math.isqrt(value.shape[0])
    if side*side != value.shape[0]:
        raise ValueError('Prediction token grid must be square')
    # Output is (p_h,p_w,p_t,C), NOT the C-first input patch order.
    return value.reshape(1, 1, side, side, 2, 2, 1, 16).transpose(0, 7, 1, 6, 2, 4, 3, 5).reshape(1, 16, 1, 2*side, 2*side)


@dataclass(frozen=True)
class Schedule:
    sigmas: np.ndarray
    timestamps: np.ndarray
    indices: np.ndarray
    times: np.ndarray
    coefficients: np.ndarray
    deltas: np.ndarray
    sin_times: np.ndarray

    @property
    def updates(self):return int(self.indices.size)

    @classmethod
    def from_arrays(cls, arrays, *, hidden=2048):
        fields = ('sigmas', 'timestamps', 'indices', 'times', 'coefficients', 'deltas', 'sin_times')
        if set(arrays) != set(fields):
            raise ValueError('Unexpected C206 schedule arrays')
        value = cls(**{name: np.array(arrays[name], copy=True) for name in fields})
        if value.indices.ndim!=1 or not value.indices.size:
            raise ValueError('A complete nonempty Euler update sequence is required')
        updates=value.updates
        expected = dict(sigmas=(updates+2,), timestamps=(updates+1,), indices=(updates,), times=(updates,),
                        coefficients=(updates,), deltas=(updates,), sin_times=(updates, 1, hidden))
        for name, shape in expected.items():
            item = getattr(value, name)
            if item.shape != shape or item.dtype != np.dtype('<i4' if name == 'indices' else '<f4'):
                raise ValueError('Invalid C206 schedule '+name)
            require_finite(item, name)
        active = np.flatnonzero(value.sigmas[1:] != value.sigmas[:-1]).astype('<i4')
        if not np.array_equal(active, value.indices) or not np.array_equal(active, np.arange(updates, dtype='<i4')):
            raise ValueError('Schedule must include every descending update and only one terminal zero update')
        if (value.sigmas[0]!=np.float32(80) or value.sigmas[-1]!=np.float32(.002)
                or value.sigmas[-2]!=value.sigmas[-1]):
            raise ValueError('Complete Cosmos sigma_max80 to sigma_min0.002 schedule required, not a truncated prefix')
        if np.any(value.sigmas <= 0) or np.any(np.diff(value.sigmas[:-1]) >= 0) or np.any(value.deltas >= 0):
            raise ValueError('Invalid positive descending sigma schedule')
        if not np.array_equal(value.timestamps,np.multiply(value.sigmas[:-1],np.float32(1000),dtype=np.float32)):
            raise ValueError('Timestamp table must match the fixed1000 training-step scheduler')
        for step, index in enumerate(value.indices):
            sigma = value.sigmas[index]
            time = np.divide(sigma, np.add(sigma, np.float32(1), dtype=np.float32), dtype=np.float32)
            coefficient = np.float32(1-float(time))
            delta = np.subtract(value.sigmas[index+1], sigma, dtype=np.float32)
            if (time.tobytes() != value.times[step].tobytes() or
                    coefficient.tobytes() != value.coefficients[step].tobytes() or
                    delta.tobytes() != value.deltas[step].tobytes()):
                raise ValueError('Schedule scalar table differs from reference operation order')
        return value

    def input_for_step(self, latent, step):
        self.check_step(step)
        return patches(np.multiply(f32(latent), self.coefficients[step], dtype=np.float32)), self.sin_times[step]

    def check_step(self, step):
        if type(step) is not int or not 0 <= step < self.updates:
            raise ValueError('Update index outside this complete schedule')

    def advance(self, latent, positive, negative, step):
        self.check_step(step)
        latent, positive, negative = map(f32, (latent, positive, negative))
        if latent.shape != positive.shape or latent.shape != negative.shape:
            raise ValueError('Euler tensors have different shapes')
        for name, item in (('latent', latent), ('positive', positive), ('negative', negative)):
            require_finite(item, name)
        scaled = np.multiply(latent, self.coefficients[step], dtype=np.float32)
        cond = np.subtract(scaled, np.multiply(positive, self.times[step], dtype=np.float32), dtype=np.float32)
        uncond = np.subtract(scaled, np.multiply(negative, self.times[step], dtype=np.float32), dtype=np.float32)
        difference = np.subtract(cond, uncond, dtype=np.float32)
        denoised = np.add(cond, np.multiply(np.float32(7), difference, dtype=np.float32), dtype=np.float32)
        flow = np.divide(np.subtract(latent, denoised, dtype=np.float32), self.sigmas[self.indices[step]], dtype=np.float32)
        result = np.add(latent, np.multiply(self.deltas[step], flow, dtype=np.float32), dtype=np.float32)
        require_finite(result, 'next latent')
        return result


def validate_reference(schedule, reference):
    """Fail closed if this CPU changes any recorded intermediate rounding."""
    if set(reference) != {'initial', 'positive', 'negative', 'expected'}:
        raise ValueError('Unexpected sampler reference arrays')
    value = f32(reference['initial'])
    if reference['positive'].shape != (schedule.updates,)+value.shape or reference['negative'].shape != (schedule.updates,)+value.shape:
        raise ValueError('Invalid sampler reference prediction shapes')
    if reference['expected'].shape != (schedule.updates,)+value.shape:
        raise ValueError('Invalid sampler reference result shapes')
    for step in range(schedule.updates):
        value = schedule.advance(value, reference['positive'][step], reference['negative'][step], step)
        if value.tobytes() != f32(reference['expected'][step]).tobytes():
            raise RuntimeError('C206 FP32 sampler differs from Diffusers/Torch reference at step%d' % step)
    return digest(value)


class TorchMT19937:
    """MT19937 initialization/tempering used by the PyTorch CPU generator.

    This is not numpy.random: NumPy's seeding and normal transform differ.
    See PyTorch aten/src/ATen/core/MT19937RNGEngine.h (BSD-3-Clause).
    """
    def __init__(self, seed):
        self.state = [int(seed) & 0xffffffff]
        for index in range(1, 624):
            prev = self.state[-1]
            self.state.append((1812433253*(prev ^ (prev >> 30))+index) & 0xffffffff)
        self.index = 624

    def next(self):
        if self.index == 624:
            for index in range(624):
                joined = (self.state[index] & 0x80000000) | (self.state[(index+1) % 624] & 0x7fffffff)
                self.state[index] = self.state[(index+397) % 624] ^ (joined >> 1) ^ (0x9908b0df if joined & 1 else 0)
            self.index = 0
        value = self.state[self.index]
        self.index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9d2c5680
        value ^= (value << 15) & 0xefc60000
        value ^= value >> 18
        return value & 0xffffffff


def portable_torch_normal(shape, seed):
    """PyTorch contiguous CPU float32 MT19937/16-value Box-Muller algorithm.

    libm/SIMD implementations can differ in their last bits. The caller MUST
    compare against the frozen reference and may never silently accept a new
    sequence. AVX compatibility is provided separately when required.
    """
    count = math.prod(shape)
    if count < 16 or count % 16:
        raise ValueError('C206 latent count must be divisible by16')
    generator = TorchMT19937(seed)
    uniform = np.fromiter(((generator.next() & 0xffffff)/16777216 for _ in range(count)), dtype='<f4', count=count)
    matrix = uniform.reshape(-1, 16)
    u1 = np.subtract(np.float32(1), matrix[:, :8], dtype=np.float32)
    # Scalar PyTorch uses logf/sqrtf/cosf/sinf and stores theta in float32.
    radius = np.sqrt(np.multiply(np.float32(-2), np.log(u1, dtype=np.float32), dtype=np.float32), dtype=np.float32)
    theta = np.asarray(matrix[:, 8:].astype(np.float64)*(2*math.pi), dtype='<f4')
    matrix[:, :8] = np.multiply(radius, np.cos(theta, dtype=np.float32), dtype=np.float32)
    matrix[:, 8:] = np.multiply(radius, np.sin(theta, dtype=np.float32), dtype=np.float32)
    return uniform.reshape(shape)


def initial_latent(shape, seed, sigma, reference, *, backend='auto', allow_reference_fallback=True):
    """Generate only on node1; explicitly record any regression-seed replay.

    The regression fixture is an INITIAL latent, never a model prediction or
    final output. Cross-architecture libm differences may require its explicit
    replay for the fixed regression seed. Other seeds always run the selected
    generator and declare its identity, without claiming PC bit equivalence.
    """
    if type(seed) is not int or not 0 <= seed <= 0xffffffff:
        raise ValueError('C206 seed must be uint32')
    used = backend
    if backend not in ('auto', 'torch', 'portable'):
        raise ValueError('Unknown initial noise backend')
    if backend in ('auto', 'torch'):
        try:
            import torch
        except ImportError:
            if backend == 'torch':
                raise RuntimeError('TorchCPU is unavailable on node1; a validated portable backend is required')
            used = 'portable'
        else:
            used = 'torch'
            noise = torch.randn(tuple(shape), dtype=torch.float32, device='cpu',
                                generator=torch.Generator(device='cpu').manual_seed(seed)).numpy()
    if used == 'portable':
        noise = portable_torch_normal(shape, seed)
    latent = np.multiply(f32(noise), np.float32(sigma), dtype=np.float32)
    info = dict(backend=used, seed=seed, generated_sha256=digest(latent),
                algorithm='torch-cpu-mt19937-normal16' if used == 'torch' else 'mt19937-normal16-numpy-f32-libm-v1',
                numpy_version=np.__version__, reference_comparison='not_applicable_to_this_seed')
    if used == 'torch':
        info.update(torch_version=torch.__version__, cpu_capability=torch.backends.cpu.get_cpu_capability())
    if seed == SEED:
        if latent.shape != reference.shape:
            raise ValueError('Initial reference shape differs from preset')
        exact = latent.tobytes() == f32(reference).tobytes()
        info.update(reference_exact=exact, reference_comparison='exact' if exact else 'different',
                    reference_max_abs_error=float(np.max(np.abs(latent-f32(reference)))))
        if not exact:
            if not allow_reference_fallback:
                raise RuntimeError('Initial generator differs from PC reference; reference replay disabled')
            latent = f32(reference).copy()
            info['backend'] = 'explicit-regression-initial-reference-replay'
            info['generated_backend'] = used
    require_finite(latent, 'generated initial noise')
    info['sha256'] = digest(latent)
    return latent, info
