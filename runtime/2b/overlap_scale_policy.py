"""SCL1 fixed AdaLN scales for the fixed 2B image.

This is an analytic range policy, not calibration or image-quality evidence.
The exact centered-variance bound is sqrt(2048). The inherited bound80 stays
conservative; this port changes shape admission, not the numerical policy.
Ordinary activations and matrix
weights keep their original formats. No weight file is rewritten here.
"""
import math
import numpy as np
from tp4_profile import PROFILE

MAGIC = 0x53434C31
VERSION = 1
POLICY_ANALYTIC_POW2 = 1
NORMALIZED_BOUND = 80.0
COEFFICIENTS = PROFILE['hidden']
VS_POLICY = 'PV_PROB_SCALE_FOLDED_V1'


def _actual_half(values):
    raw = np.asarray(values, dtype='<f4').reshape(-1)
    if raw.size != COEFFICIENTS or not np.isfinite(raw).all():
        raise ValueError(f'SCL1 requires {COEFFICIENTS} finite modulation coefficients')
    bits = raw.view('<u4')
    magnitude = bits & np.uint32(0x7fffffff)
    with np.errstate(over='ignore', under='ignore'):
        encoded = raw.astype('<f2').view('<u2').copy()
    sign = ((bits >> 16) & np.uint32(0x8000)).astype('<u2')
    encoded[magnitude <= 0x387fe000] = sign[magnitude <= 0x387fe000]
    edge = (magnitude > 0x387fe000) & (magnitude < 0x38800000)
    encoded[edge] = sign[edge] | np.uint16(0x0400)
    result = encoded.view('<f2').astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError('SCL1 modulation coefficients overflow binary16')
    return result


def build_scale_record(shift, gain):
    """Return eight U32 words; callers allocate/flush one 32-byte record.

    Bound each rounded half multiply/add, then round the common scale upward
    to a power of two. The normalized bound is intentionally conservative;
    image quality across seeds is unvalidated, not inferred from the bound.
    """
    shift_half, gain_half = _actual_half(shift), _actual_half(gain)
    half_margin = 1.0 + 2.0 ** -10
    bound = float(np.max((NORMALIZED_BOUND * np.abs(gain_half) * half_margin
                          + np.abs(shift_half)) * half_margin))
    if not math.isfinite(bound) or bound > 65504.0:
        raise ValueError('SCL1 conservative modulated range exceeds finite binary16')
    exponent = math.ceil(math.log2(bound / 32760.0)) if bound else 0
    scale, inverse = math.ldexp(1.0, exponent), math.ldexp(1.0, -exponent)
    result = np.zeros(8, dtype='<u4')
    result[:2] = MAGIC, VERSION
    result[2:5] = np.array([scale, inverse, bound], dtype='<f4').view('<u4')
    result[5] = POLICY_ANALYTIC_POW2
    return result


def fold_v_scales(values):
    """Produce new FP32 VS values; never modify a saved asset in place."""
    source = np.asarray(values, dtype='<f4')
    if not np.isfinite(source).all() or np.any(source <= 0):
        raise ValueError('Expected finite positive original V scales')
    return np.multiply(source, np.float32(1.0 / 32760.0), dtype=np.float32)
