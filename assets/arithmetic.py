"""Arithmetic and packing used by the shared five-slice denoiser."""
import numpy as np

def round16(value):
    value=np.asarray(value,dtype='<f4')
    bits=value.view('<u4'); magnitude=bits&0x7fffffff; sign=bits&0x80000000
    with np.errstate(over='ignore',under='ignore'):
        result=value.astype('<f2').astype('<f4')
    encoded=result.view('<u4').copy()
    encoded=np.where(magnitude<=0x387fe000,sign,encoded).astype('<u4')
    encoded=np.where((magnitude>0x387fe000)&(magnitude<0x38800000),sign|0x38800000,encoded).astype('<u4')
    encoded=np.where((encoded&0x7fffffff)<0x38800000,encoded&0x80000000,encoded).astype('<u4')
    return encoded.view('<f4')


def quantize(value):
    value=round16(value); maximum=np.max(np.abs(value),axis=-1)
    scale=np.where(maximum==0,np.float32(1),maximum/np.float32(32760)).astype('<f4')
    inverse=np.float32(1)/scale
    scaled=np.multiply(value,inverse[...,None],dtype=np.float32)
    shifted=np.add(scaled,np.where(scaled>=0,np.float32(.5),np.float32(-.5)),dtype=np.float32)
    return np.clip(np.trunc(shifted),-32760,32760).astype('<i2'),scale


def silu(value,table):
    value=round16(value);negative=(value.view('<u4')>>31)!=0;magnitude=np.abs(value)
    active=magnitude<np.float32(12)
    scaled=round16(np.where(active,magnitude,np.float32(0))*np.float32(32))
    index=scaled.astype(np.uint32);fraction=round16(scaled-index.astype('<f4'))
    coefficient=table[index]
    base=(coefficient&65535).astype('<u2').view('<f2').astype('<f4')
    change=(coefficient>>16).astype('<u2').view('<f2').astype('<f4')
    base=round16(base);change=round16(change)
    correction=round16(base+round16(change*fraction))
    positive=round16(value-correction)
    negative_value=(correction.view('<u4')^0x80000000).view('<f4')
    return np.where(active,np.where(negative,negative_value,positive),np.where(negative,np.float32(0),value)).astype('<f4')


def finite(value,label):
    value=np.asarray(value)
    if value.dtype.hasobject or (value.dtype.kind in 'fc' and not np.isfinite(value).all()):raise ValueError('Nonfinite/object tensor: '+label)
    return value

def linear(value,weight,weight_scale):
    value=finite(np.asarray(value,dtype='<f4'),'linear input')
    weight=np.asarray(weight);weight_scale=finite(np.asarray(weight_scale,dtype='<f4'),'weight scale')
    if value.ndim!=2 or weight.ndim!=2 or weight.dtype!=np.dtype('<i2') or weight.shape[1]!=value.shape[1] or weight_scale.shape!=(weight.shape[0],):
        raise ValueError('W16 linear shape/dtype mismatch')
    if not 1<=value.shape[1]<=8192:raise ValueError('INT48 accumulation bound')
    if np.any(weight_scale<=0):raise ValueError('Weight scales must be positive')
    q,scale=quantize(value);out=np.empty((len(q),len(weight)),dtype='<f4')
    # Every INT16 product and every partial sum is an integer of magnitude
    # <=8192*32768^2=2^43. IEEE binary64 represents all of these exactly, so
    # a BLAS double dot (including FMA/reassociation) is identical to INT48.
    # The result only enters FP32 after the entire K reduction, as in PL.
    q64=q.astype(np.float64)
    for first in range(0,len(weight),512):
        stop=min(first+512,len(weight))
        acc=q64@np.asarray(weight[first:stop],dtype=np.float64).T
        if np.any(np.abs(acc)>2**43) or np.any(acc!=np.trunc(acc)):
            raise ArithmeticError('Exact bounded integer dot violated')
        x=np.multiply(acc.astype('<f4'),scale[:,None],dtype=np.float32)
        out[:,first:stop]=round16(np.multiply(x,weight_scale[None,first:stop],dtype=np.float32))
    return finite(out,'linear output')

def rms(value,gamma):
    value=round16(finite(np.asarray(value,dtype='<f4'),'RMS input'))
    if value.ndim!=2 or value.shape[1]%128:raise ValueError('RMS width must contain128-value phase laps')
    gamma=finite(np.asarray(gamma,dtype='<f4'),'RMS gamma')
    if gamma.shape!=(value.shape[1],):raise ValueError('RMS gamma width')
    phases=np.zeros((len(value),16,8),dtype='<f4')
    centered=np.subtract(value,np.float32(0),dtype=np.float32)
    squares=np.multiply(centered,centered,dtype=np.float32)
    for k in range(0,value.shape[1],128):phases=np.add(phases,squares[:,k:k+128].reshape(len(value),16,8),dtype=np.float32)
    while phases.shape[1]>1:phases=np.add(phases[:,0::2,:],phases[:,1::2,:],dtype=np.float32)
    lanes=phases[:,0,:]
    while lanes.shape[1]>1:lanes=np.add(lanes[:,0::2],lanes[:,1::2],dtype=np.float32)
    variance=np.divide(lanes[:,0],np.float32(value.shape[1]),dtype=np.float32)
    inv=np.divide(np.float32(1),np.sqrt(np.add(variance,np.float32(1e-6),dtype=np.float32)),dtype=np.float32)
    normalized=np.multiply(np.multiply(centered,inv[:,None],dtype=np.float32),gamma,dtype=np.float32)
    return round16(np.add(normalized,np.float32(0),dtype=np.float32))

def add(a,b):return round16(np.add(round16(a),round16(b),dtype=np.float32))

def packed_kv(keys,values,heads=10):
    keys=round16(finite(keys,'cross keys'));values=round16(finite(values,'cross values'))
    if keys.ndim!=2 or keys.shape!=values.shape or keys.shape[1]!=heads*128 or not 1<=len(keys)<=512:
        raise ValueError('Cross-KV geometry')
    stored=len(keys);blocks=(stored+127)//128;stride=(heads+7)//8*8
    k,scales=quantize(keys.reshape(stored,heads,128));ks=np.zeros((stored,stride),dtype='<f4');ks[:,:heads]=scales
    v=np.zeros((heads,blocks,128,128),dtype='<i2');vs=np.empty((heads,blocks,128),dtype='<f4')
    for block in range(blocks):
        start=block*128;count=min(128,stored-start)
        data=values[start:start+count].reshape(count,heads,128).transpose(1,2,0)
        quantized,scale=quantize(data);v[:,block,:,:count]=quantized;vs[:,block]=scale
    return dict(k=np.ascontiguousarray(k.reshape(stored,heads*128)),v=v,ks=ks,vs=vs)

