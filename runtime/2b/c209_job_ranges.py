"""Conservative V2 execution footprints and alias checks, mirrored in C++.

Footprints include each final 32-byte AXI beat and intervening row padding.
This is separate from ABI serialization: decoder-only fixtures may contain
arbitrary addresses. Operation shape validation remains a separate gate.
"""
from c209_program import validate_job

_BYTES={1:2,2:4,3:2,4:2,5:4,6:1}


def validate_descriptor_span(proofs,address,record_count,record_bytes=128):
    """Prevent tensor writes from modifying the program being consumed."""
    if (type(address) is not int or not 0<=address<1<<64 or address&31
            or type(record_count) is not int or not 1<=record_count<=128):
        raise ValueError('invalid V2 descriptor storage address/count')
    if record_bytes not in (64,128):raise ValueError('Unknown descriptor record size')
    end=address+record_count*record_bytes
    if end>=1<<64:
        raise ValueError('V2 descriptor storage exceeds uint64 exclusive-end representation')
    for proof in proofs:
        for span in proof['ranges']:
            if span['write'] and span['start']<end and address<span['end']:
                raise ValueError('tensor write overlaps active V2 descriptor program: '+span['slot'])
    return dict(start=address,end=end,record_count=record_count)


def validate_job_ranges(job):
    j=validate_job(job); ranges=[]
    m,n,k=j['rows'],j['cols'],j['inner']
    la,lb,lc=j['lda'],j['ldb'],j['ldc']
    stage,mode,flags=j['stage'],j['mode'],j['flags']

    def span(slot,count,write=False,*,rows=1,cols=None,stride=0):
        if not count:return
        start=j[slot];end=start+count
        if end>=1<<64:raise ValueError('tensor access exceeds uint64 exclusive-end representation: '+slot)
        ranges.append(dict(slot=slot,start=start,end=end,write=write,rows=rows,
                           cols=cols,stride=stride,format=j['formats'][slot]))

    def round32(count):return (count+31)//32*32

    def vector(slot,count,write=False):
        if count:span(slot,round32(count),write)

    def matrix(slot,rows,cols,stride,write=False,bytes_per_element=None):
        if not rows or not cols:return
        size=bytes_per_element or _BYTES[j['formats'][slot]]
        if stride<cols or stride%(32//size):
            raise ValueError('invalid accessed columns or AXI-aligned row stride: '+slot)
        span(slot,(rows-1)*stride*size+round32(cols*size),write,
             rows=rows,cols=cols,stride=stride)

    def shape(condition):
        if not condition:raise ValueError('invalid stage dimensions/K window/prepared length')

    if stage in (1,15,16):
        shape(1<=m<=4096 and 1<=n<=8192 and 1<=k<=8192)
        end=j['k_start']+j['k_count'] if j['k_count'] else k
        shape(j['k_start']%16==0 and j['k_start']<end<=k)
        matrix('a',m,k,la)
        if flags&64:
            shape(j['formats']['b']==3 and k%16==0 and n%16==0 and lb==k and not j['k_start'] and not j['k_count'])
            physical_k=64 if stage==1 and k==80 else k
            span('b',n*physical_k*2,rows=n,cols=physical_k,stride=0)
        else:matrix('b',n,k,lb)
        matrix('c',m,n,lc,True)
        vector('aux0',m*4);vector('aux1',n*4)
        if stage==15:vector('aux2',m*4,True)
        if flags&1:vector('aux2',n*4)
        if flags&16:vector('aux3',32)
    elif stage in (2,8):
        shape(1<=m<=4096 and 1<=n<=8192 and lc>=n)
        matrix('a',m,n,la);matrix('c',m,lc,lc,True);vector('aux0',m*4,True)
    elif stage in (3,4,10):
        shape(1<=m<=(4096 if stage==10 else 65536) and 1<=n<=8192)
        matrix('a',m,n,la);vector('aux0',n*4)
        if stage==10 or flags&1:vector('aux1',n*4)
        if stage==10:vector('aux2',n*4)
        if stage==10 and not mode&2:
            shape(lc>=n)
            matrix('c',m,lc,lc,True);vector('aux3',m*4,True)
        else:matrix('c',m,n,lc,True)
    elif stage==5:
        shape(1<=m<=4096 and 1<=n<=8192)
        matrix('a',m,n,la);matrix('c',m,n,lc,True)
        if mode in (0,1,3,7):matrix('b',m,n,lb)
        if mode in (2,3,7):vector('aux0',n*4)
        if mode==2:vector('aux1',n*4)
        if mode==7:matrix('aux1',m,n,lc)
    elif stage==6:
        shape(1<=m<=4096 and 1<=j['heads']<=16 and n==j['heads']*128)
        width=j['heads']*128
        matrix('a',m,width,la);matrix('c',m,width,lc,True)
        vector('aux0',m*128*4);vector('aux1',m*128*4)
    elif stage==9:
        shape(1<=m<=4096 and 1<=n<=8192 and lc>=n)
        matrix('a',m,n,la);vector('aux0',m*4)
        matrix('c',m,min(lc,(n+15)//16*16),lc,True)
    elif stage in (7,12):
        heads=j['heads'];width=heads*128
        shape(1<=n<=4096 and 1<=heads<=16)
        if stage==12:
            shape(m<=n and (j['k_count']==m if flags&32 else m==n))
        else:shape(1<=m<=4096)
        stored=m if stage==12 else (j['k_count'] if flags&32 else n)
        shape(stored<=n)
        ab=_BYTES[j['formats']['a']]
        bb=_BYTES[j['formats']['b']]
        cb=_BYTES[j['formats']['c']]
        shape(la>=width and lb>=width and lc>=width and la%(32//ab)==0
              and lb%(32//bb if stage==12 else 16)==0
              and lc%(16 if stage==12 else 32//cb)==0)
        blocks=(stored+127)//128;head_stride=(heads+7)//8*8
        if stage==7:
            matrix('c',m,width,lc,True)
            if stored:
                matrix('a',m,width,la);matrix('b',stored,width,lb,bytes_per_element=2)
        elif stored:
            matrix('a',m,width,la);matrix('b',m,width,lb)
            matrix('c',m,width,lc,True,bytes_per_element=2)
        if stored:
            write=stage==12
            vector('aux0',stored*head_stride*4,write)
            vector('aux1',heads*blocks*128*4,write)
            vector('aux2',heads*blocks*32768,write)
    elif stage==13:
        heads=j['heads'];start=j['k_start'];width=heads*128
        shape(heads==8 and n==4096 and m in (2048,4096) and start%128==0 and start+m<=n and j['k_count']==m)
        matrix('a',m,width,la);matrix('b',m,width,lb)
        # Conservative global destination envelopes include head-major V/VS
        # gaps, although the PL writes only this group's exact token slice.
        vector('c',n*width*2,True);vector('aux0',n*8*4,True)
        vector('aux1',heads*(n//128)*128*4,True);vector('aux2',heads*(n//128)*32768,True)
    elif stage==14:
        shape(m in (2048,4096) and n==2048 and la==lc==2048)
        matrix('a',m,n,la);matrix('c',m,n,lc,True)
        vector('aux0',32);vector('aux1',n*4);vector('aux2',n*4);vector('aux3',m*4,True)
    elif stage==17:
        shape(m in (2048,4096) and n==1024 and j['heads']==8 and la==lc==1024)
        matrix('a',m,n,la);matrix('c',m,n,lc,True)
        vector('aux0',128*4);vector('aux1',m*128*4);vector('aux2',m*128*4)

    aliases=[]
    def exact_inplace(write,read):
        if write['slot']!='c':return False
        allowed=(stage==5 and (read['slot'] in ('a','b') or read['slot']=='aux1' and mode==7)) or (
            (stage in (3,4,6,17) or stage==10 and mode&2) and read['slot']=='a')
        if not allowed:return False
        return all(write[key]==read[key] for key in ('start','end','rows','cols','stride','format'))

    for index,left in enumerate(ranges):
        for right in ranges[index+1:]:
            if not (left['write'] or right['write']) or left['end']<=right['start'] or right['end']<=left['start']:
                continue
            if left['write'] and right['write']:
                raise ValueError('overlapping tensor writes: '+left['slot']+'/'+right['slot'])
            write,read=(left,right) if left['write'] else (right,left)
            if not exact_inplace(write,read):
                raise ValueError('unsupported tensor read/write overlap: '+write['slot']+'/'+read['slot'])
            aliases.append([write['slot'],read['slot']])
    return dict(kind='C209_MODULAR16_EXECUTION_RANGES_V1',ranges=ranges,
                exact_inplace_aliases=aliases,beat_bytes=32,stage_geometry_checked=True,
                all_operation_semantics_proved=False)
