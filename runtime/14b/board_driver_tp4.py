"""One identical image for all TP ranks/stages. Real native MMIO/DMA only."""
import ctypes as C
import json
import hashlib
import re
import os
import threading
import time
from collections import OrderedDict
import numpy as np
from pathlib import Path
from board_driver import CoreJob,Stats,RegWrite
from fixed_native_plan import PreparedCommand,CoreCompletion,COUNTER_NAMES,projection_type
from board_driver_c206 import C206Board,StreamLane,StreamReduceStats,BulkJob,acquire_hardware_lease,NEIGHBOURS
from board_driver_c209 import C209Board
from tp4_profile import PROFILE,BIT_ID,RUNTIME_ID,Role,condition_id,validate_job,validate_generation,EPOCHS_PER_GENERATION
from sp3_control import COMMUNICATION_PROTOCOL,NATIVE_CHUNK_PIPELINE_REVISION,NATIVE_FIXED_TP_CONTROL_MODE

class PairJob(C.Structure):
    _fields_=[('lane',StreamLane)]+[(n,C.c_uint64) for n in ('local_phys','output_phys','residual_phys','gate_phys')]+[(n,C.c_uint32) for n in
        ('rows','cols','first_row','node','peer','lane_id','fused','epoch','step','condition','layer','operator_id','timeout_ms')]

NativeProjection=projection_type(PairJob,StreamReduceStats)

class SpanJob(C.Structure):
    _fields_=[('lane',StreamLane)]+[(n,C.c_uint64) for n in ('tensor_phys','dummy_tx_phys','dummy_rx_phys')]+[(n,C.c_uint32) for n in
        ('bytes','node','peer','lane_id','epoch','step','channel','timeout_ms','sending')]

class DuplexSpanJob(C.Structure):
    _fields_=[('lane',StreamLane)]+[(n,C.c_uint64) for n in ('tx_phys','rx_phys','dummy_tx_phys','dummy_rx_phys')]+[(n,C.c_uint32) for n in
        ('tx_bytes','rx_bytes','node','peer','lane_id','epoch','step','channel','timeout_ms')]

def wire_epoch(generation,stage,step,condition,layer=0,operator=0):
    validate_generation(generation)
    if not 0<=stage<PROFILE['sequence_parallel']:raise ValueError('Sequence group')
    if not 0<=step<PROFILE['updates']:raise ValueError('Step')
    if not 0<=layer<PROFILE['layers']:raise ValueError('Layer')
    if type(operator)is not int or not 0<=operator<3:raise ValueError('Operator')
    return generation*EPOCHS_PER_GENERATION+stage*65536+step*1024+layer*8+operator*2+condition_id(condition)

class TP4Board(C209Board):
    def __init__(self,root,role,*,load=False,rx_polarities=None):
        if not isinstance(role,Role):raise TypeError('Explicit immutable role required')
        self.root=Path(root);self.assignment=role;self.node=role.node;self.rank=role.rank;self.group=role.stage
        self.role='COMMON';self.peer=(role.rank^1)+1;self.condition=None
        self.buffers=[];self.program_buffer=None;self.weight_region=None;self.running=False
        self.poisoned=False;self.needs_reinitialization=False;self._init_concurrency();self._lane_locks=[threading.Lock() for _ in range(4)]
        self._last_pair_context={};self._last_bulk_context={}
        self.protocol=json.loads((self.root/'protocol.json').read_text())
        from five_fixed_jobs import require_image_contract
        require_image_contract(self.protocol)
        if (self.protocol.get('sum_only_reducer_ports')!=[0] or self.protocol.get('fused_residual_ports')!=[1]
            or self.protocol.get('native_sum_only_mask')!=1):raise ValueError('Fixed P0 SUM/P1 FUSED runtime package required')
        if self.protocol.get('profile_sha256')!=PROFILE['profile_sha256'] or self.protocol.get('hardware_id')!=PROFILE['hardware_id']:
            raise ValueError('Package TP4 identity/geometry mismatch')
        if not self.protocol.get('bitstream_generated'):raise RuntimeError('This synthesis-only candidate has no deployable image')
        if not self.protocol.get('control_header_verified'):raise ValueError('Generated register header required')
        header=(self.root/'xcosmos2b_tile_hw.h').read_bytes()
        if hashlib.sha256(header).hexdigest()!=self.protocol['control_header_sha256']:raise ValueError('Generated header mismatch')
        generated={m.group(1).decode().lower():int(m.group(2),16) for m in re.finditer(rb'#define\s+XCOSMOS2B_TILE_CONTROL_ADDR_(\w+)_DATA\s+(0x[0-9a-fA-F]+)',header)}
        self.regs=self.protocol['registers']
        if any(generated.get(k)!=v for k,v in self.regs.items()):raise ValueError('Actual CSR offsets differ')
        self.program_enabled=True
        if rx_polarities is None or len(rx_polarities)!=4 or any(type(x)is not int or x not in (0,1) for x in rx_polarities):raise ValueError('Four explicit physical-board lane polarities required; never infer from rank')
        bit=self.root/'hardware/cosmos_net.bit'
        if not bit.is_file() or not bit.with_suffix('.hwh').is_file():raise FileNotFoundError('Common bit/HWH missing')
        self._hardware_lease=acquire_hardware_lease('board')
        try:self._open_tp4(bit,load,rx_polarities)
        except BaseException:self._hardware_lease.close();self._hardware_lease=None;raise
    def _open_tp4(self,bit,load,polarities):
        os.environ.setdefault('BOARD','ZCU102')
        from pynq import Overlay,MMIO,Clocks,allocate
        class RuntimeOverlay(Overlay):
            def _register_drivers(self):pass
        self.allocate=allocate;self.overlay=RuntimeOverlay(str(bit),download=load)
        self.core=MMIO(self.protocol['base'],4096);self.clock=float(Clocks.fclk0_mhz)
        if not self.core.read(0)&4:raise RuntimeError('Core already active')
        contract=self.protocol.get('tp_batch_controller',{})
        if (contract.get('required') is not True or contract.get('base')!=0xA0210000 or contract.get('identity')!=0xC209B001
            or contract.get('revision')!=1 or contract.get('expected_bit_id')!=BIT_ID):raise ValueError('Corrected fixed FP32 batch controller and actual HWH BIT_ID contract required')
        self.batch_controller=MMIO(contract['base'],4096)
        if self.batch_controller.read(0)!=contract['identity']:raise RuntimeError('Missing fixed FP32 batch controller')
        if self.batch_controller.read(4)&1:raise RuntimeError('Batch controller still owns DMA CSRs')
        self.batch_controller.write(8,2)
        self.interfaces=[];self._attach_counter_history=[]
        entries=self.protocol['network_interfaces']
        if sorted(e['lane'] for e in entries)!=[0,1,2,3]:raise ValueError('Four actual interfaces required')
        for entry in sorted(entries,key=lambda e:e['lane']):
            i=entry['lane']
            if entry.get('sum_only')!=(i==0) or entry.get('bulk_only')!=(i>=2):raise ValueError('Reducer capability differs from the fixed physical port')
            residual=entry.get('residual_dma_base')
            if (i==1 and not residual) or (i!=1 and residual is not None):raise ValueError('Residual DMA must exist only on P1')
            lane=dict(lane=i,link=MMIO(entry['link_base'],4096),dma=MMIO(entry['tx_dma_base'],65536),reducer=MMIO(entry['reducer_base'],4096))
            lane['residual_dma']=MMIO(residual,65536) if i==1 else None
            if lane['link'].read(0)!=0xc206e001 or lane['link'].read(0x7c)!=BIT_ID or lane['reducer'].read(0)!=0xc206f001:raise RuntimeError('Wrong lane image')
            self._attach_counter_history.append(dict(port=i,status=int(lane['link'].read(4)),rx_errors=int(lane['link'].read(0x4c)),mac_faults=int(lane['link'].read(0x50)),router_drops=int(lane['link'].read(0xd4))))
            lane['link'].write(4,2);lane['link'].write(8,(self.rank&1)|(polarities[i]<<1))
            wire_node=self.node if i<2 else self.group+1
            wire_peer=((self.rank^(1<<i))+1) if i<2 else ((self.group+(1 if i==2 else 2))%3+1)
            lane['link'].write(0x90,wire_node)
            lane['link'].write(0x94,wire_peer)
            devices=[(lane['dma'],(0,0x30))]+([(lane['residual_dma'],(0,))] if lane['residual_dma'] is not None else [])
            for dma,channels in devices:
                dma.write(0,4);end=time.monotonic()+5
                while dma.read(0)&4:
                    if time.monotonic()>end:raise TimeoutError('Owned DMA reset')
                for off in channels:dma.write(off,1);dma.write(off+4,0x1000)
            self.interfaces.append(lane)
        self.link,self.dma=self.interfaces[0]['link'],self.interfaces[0]['dma']
        entries=sorted(entries,key=lambda e:e['lane'])
        self._batch_bases=(C.c_uint32*7)(entries[0]['link_base'],entries[0]['tx_dma_base'],entries[0]['reducer_base'],entries[1]['link_base'],entries[1]['tx_dma_base'],entries[1]['reducer_base'],entries[1]['residual_dma_base'])
        self.lib=C.CDLL(str(self.root/'libcosmos_tp4.so'))
        self.lib.cosmos_tp4_runtime_identity.restype=C.c_uint32
        if self.lib.cosmos_tp4_runtime_identity()!=RUNTIME_ID:raise RuntimeError('Wrong native image')
        shell_contract=getattr(self.lib,'cosmos_tp4_sum_only_mask',None)
        if shell_contract is None:raise RuntimeError('Native executor predates P0 residual-DMA removal; rebuild this package executor')
        shell_contract.restype=C.c_uint32
        if shell_contract()!=1:raise RuntimeError('Native executor has a different reducer capability mask')
        for name,job,stat in (('cosmos_stage',CoreJob,Stats),('cosmos_tp4_pair',PairJob,StreamReduceStats),('cosmos_tp4_bulk',BulkJob,StreamReduceStats),('cosmos_tp4_bulk_span',SpanJob,StreamReduceStats),('cosmos_tp4_bulk_duplex_span',DuplexSpanJob,StreamReduceStats)):
            f=getattr(self.lib,name);f.argtypes=[C.POINTER(job),C.POINTER(stat)];f.restype=C.c_int
        for name,typ in (('cosmos_core_job_size',CoreJob),('cosmos_stats_size',Stats),('cosmos_tp4_pair_job_size',PairJob),('cosmos_tp4_span_job_size',SpanJob),('cosmos_tp4_duplex_span_job_size',DuplexSpanJob),('cosmos_stream_stats_size',StreamReduceStats),('cosmos_c206_bulk_job_size',BulkJob)):
            f=getattr(self.lib,name);f.restype=C.c_size_t
            if f()!=C.sizeof(typ):raise RuntimeError('Native ABI mismatch '+name)
        revision=getattr(self.lib,'cosmos_tp4_chunk_pipeline_revision',None)
        if revision is None:raise RuntimeError('Native executor lacks the chunk pipeline; rebuild the matching software package')
        revision.restype=C.c_uint32
        actual_revision=revision()
        if actual_revision!=NATIVE_CHUNK_PIPELINE_REVISION:raise RuntimeError('Unsupported native chunk pipeline revision')
        # Publish only after this driver checked the actually loaded library.
        # The SP3 twelve-role permit binds this value across every board.
        self.communication_protocol=COMMUNICATION_PROTOCOL
        self.native_chunk_pipeline_revision=actual_revision
        self.lib.cosmos_tp4_pair_pipeline.argtypes=[C.POINTER(PairJob),C.POINTER(PairJob),C.POINTER(StreamReduceStats)]
        self.lib.cosmos_tp4_pair_pipeline.restype=C.c_int
        for name in ('cosmos_tp4_batch_revision','cosmos_tp4_duplex_kv_revision'):
            fn=getattr(self.lib,name);fn.restype=C.c_uint32
            if fn()!=1:raise RuntimeError('Native fixed FP32 batch/duplex revision mismatch')
        self.native_batch_revision=1
        self.lib.cosmos_fixed_tp_control_mode.restype=C.c_uint32
        self.native_fixed_tp_control_mode=int(self.lib.cosmos_fixed_tp_control_mode())
        if self.native_fixed_tp_control_mode!=NATIVE_FIXED_TP_CONTROL_MODE:raise RuntimeError('This software requires the corrected hardware TP batch control mode')
        self._native_projection_receipts={}
        self.lib.cosmos_tp4_pair_batch.argtypes=[C.c_uint64,C.POINTER(PairJob),C.POINTER(PairJob),C.POINTER(C.c_uint32),C.POINTER(StreamReduceStats)]
        self.lib.cosmos_tp4_pair_batch.restype=C.c_int
        self.lib.cosmos_c209_serial_reset.argtypes=[C.c_uint64,C.c_uint64];self.lib.cosmos_c209_serial_reset.restype=C.c_int
        if self.lib.cosmos_c209_serial_reset(self.core.array.ctypes.data,1):raise RuntimeError('Serial owner initialization')
        self.lib.cosmos_tp4_cancel.argtypes=[C.c_uint32];self.lib.cosmos_tp4_cancel.restype=None;self.lib.cosmos_tp4_cancel(0)
        self._bulk_buffers={i:(self.make((8192,),'u1'),self.make((8192,),'u1')) for i in range(4)}
        required_plan=dict(revision=1,descriptor_bytes=8192,slots_per_branch=2,native_counter_snapshot=8,completion='FRESH_COMPUTE_DONE_THEN_TP_TWO_ROUND_DRAIN')
        if self.protocol.get('fixed_execution_plan')!=required_plan:raise RuntimeError('Package omits required native fixed execution plan ABI')
        self.lib.cosmos_fixed_plan_revision.restype=C.c_uint32
        if self.lib.cosmos_fixed_plan_revision()!=1:raise RuntimeError('Fixed native execution plan revision')
        for name,typ in (('cosmos_fixed_completion_size',CoreCompletion),('cosmos_fixed_projection_size',NativeProjection)):
            fn=getattr(self.lib,name);fn.restype=C.c_size_t
            if fn()!=C.sizeof(typ):raise RuntimeError('Fixed native plan ABI '+name)
        self.lib.cosmos_fixed_stage.argtypes=[C.POINTER(CoreJob),C.POINTER(C.c_uint32),C.POINTER(CoreCompletion)]
        self.lib.cosmos_fixed_stage.restype=C.c_int
        self.lib.cosmos_fixed_project_cancel.argtypes=[C.POINTER(NativeProjection),C.c_int];self.lib.cosmos_fixed_project_cancel.restype=C.c_int
        for name in ('cosmos_fixed_project_compute','cosmos_fixed_project_reduce'):
            fn=getattr(self.lib,name);fn.argtypes=[C.POINTER(NativeProjection)];fn.restype=C.c_int
        names=('perf_elapsed_cycles','perf_owner_a_cycles','perf_owner_b_cycles','perf_source_empty_cycles','perf_result_stall_cycles','perf_a_commits','perf_b_commits','perf_no_request_cycles')
        self._fixed_counter_regs=(C.c_uint32*8)(*(self.regs[n] for n in names))
        self._installed_plan=None

    def _program_envelope(self,count,address):return dict(stage=11,rows=count,aux0=address,mode=BIT_ID,heads=self.node)
    def program(self,jobs):
        if isinstance(jobs,PreparedCommand):return self._run_prepared(jobs)
        jobs=list(jobs)
        with self._program_lock:
            self._check_available()
            # A legacy/list launch may replace any descriptor bytes. Never
            # retain the installed-plan certificate across that writer.
            self._installed_plan=None
            if not self.program_enabled:raise RuntimeError('Fixed descriptor program is disabled')
            from five_program import FixedProgramCache,RECORD_BYTES,MAX_JOBS
            from c209_typed_job import buffer_metadata
            if not hasattr(self,'_fixed_programs'):self._fixed_programs=FixedProgramCache()
            prepared=self._fixed_programs.get(jobs)
            if self.program_buffer is None:self.program_buffer=self.make((RECORD_BYTES*MAX_JOBS,),'u1')
            dtype,address,capacity=buffer_metadata(self.program_buffer)
            if dtype!=np.dtype('u1') or len(self.program_buffer.shape)!=1:raise ValueError('Fixed descriptor buffer must be contiguous bytes')
            # Sixteen fixed 512-byte slots use the existing 8 KiB program
            # allocation: no extra CMA, guards or lifetime owner are added.
            # The program lock holds through native completion, so an evicted
            # descriptor has no outstanding PL reader. Long programs use the
            # original full allocation and invalidate all short-program slots.
            slot_bytes=512;slot_count=RECORD_BYTES*MAX_JOBS//slot_bytes
            if getattr(self,'_fixed_descriptor_base',None)!=address:
                self._fixed_descriptor_base=address
                self._fixed_descriptor_slots=OrderedDict()
                self._fixed_descriptor_large=None
            slots=self._fixed_descriptor_slots
            payload=prepared.payload
            if len(payload)<=slot_bytes:
                slot=slots.get(payload)
                if slot is None:
                    self._fixed_descriptor_large=None
                    if len(slots)==slot_count:_,slot=slots.popitem(last=False)
                    else:slot=next(i for i in range(slot_count) if i not in slots.values())
                    descriptor_address=address+slot*slot_bytes
                    prepared.bind_descriptor(descriptor_address,slot_bytes)
                    first=slot*slot_bytes
                    self.program_buffer.array[first:first+len(payload)]=np.frombuffer(payload,dtype='u1')
                    self.program_buffer.root.flush()
                    slots[payload]=slot
                else:
                    descriptor_address=address+slot*slot_bytes
                    prepared.bind_descriptor(descriptor_address,slot_bytes)
                    slots.move_to_end(payload)
            else:
                descriptor_address=address
                prepared.bind_descriptor(descriptor_address,capacity)
                if self._fixed_descriptor_large!=payload:
                    slots.clear()
                    self.program_buffer.array[:len(payload)]=np.frombuffer(payload,dtype='u1')
                    self.program_buffer.root.flush()
                    self._fixed_descriptor_large=payload
            result=C206Board.invoke(self,**self._program_envelope(prepared.count,descriptor_address))
            metrics=self.performance_snapshot()
            if not hasattr(self,'program_metrics'):self.program_metrics={}
            key=','.join(str(j['stage']) for j in jobs)
            entry=self.program_metrics.setdefault(key,dict(programs=0,native_ms=0.0,saturated_programs=0,
                cycles={name:0 for name in metrics['cycles']}))
            entry['programs']+=1;entry['native_ms']+=result['ms']
            entry['saturated_programs']+=int(metrics['saturated'])
            for name,value in metrics['cycles'].items():entry['cycles'][name]+=value
            result['performance']=metrics
            return result
    def bind_fixed_plan(self,plan):
        # Address geometry and ctypes structures are prepared before RELEASE.
        # This allocates only the already budgeted 8 KiB descriptor allocation.
        with self._program_lock:
            self._check_available()
            if self.program_buffer is None:self.program_buffer=self.make((8192,),'u1')
            binding=(id(self),self.program_buffer.address)
            if getattr(plan,'native_binding',None)==binding:return
            if len(plan.payload)>self.program_buffer.nbytes:raise ValueError('Fixed descriptor plan capacity')
            plan.bind(self.program_buffer.address)
            for command in plan._unique.values():
                address=self.program_buffer.address+command.offset
                fields=dict(act_offset=0,weights_offset=0,output_r_offset=0,aux_offset=0,rows=len(command),aux0=address)
                writes=[]
                for name,value in fields.items():
                    writes.append((self.regs[name],value&0xffffffff))
                    if name in self.protocol['pointer_registers']:writes.append((self.regs[name]+4,value>>32))
                settings=(RegWrite*len(writes))(*(RegWrite(*v) for v in writes))
                command.native=CoreJob(self.core.array.ctypes.data,settings,len(writes),self.regs['error'],self.regs['completed_tiles'],120000)
                command._settings=settings
            plan.native_binding=binding

    def install_fixed_plan(self,plan):
        """Caller has joined the previous layer's compute and TP readers."""
        with self._program_lock:
            self.bind_fixed_plan(plan)
            if self._installed_plan is plan:return
            self.program_buffer.array[:len(plan.payload)]=np.frombuffer(plan.payload,dtype='u1')
            self.program_buffer.root.flush()
            self._installed_plan=plan
            self._fixed_descriptor_base=None

    def _fixed_result(self,command,completion):
        values=dict(zip(COUNTER_NAMES,map(int,completion.cycles)))
        if sum(values[n] for n in COUNTER_NAMES[1:])!=values['elapsed']:raise RuntimeError('Native counter partition')
        metrics=dict(schema='FAST_GROUP_V1',clock_domain='mac_fast_group',target_mhz=300,physical_shards=5,
            cycles=values,saturated=values['elapsed']==0xffffffff,clipped=values['elapsed']==0xffffffff,
            maximum_counter_interval_seconds=0xffffffff/300000000,snapshot_scope='LAST_COMPLETED_PROGRAM',not_dsp_utilization_percent=True)
        if not hasattr(self,'program_metrics'):self.program_metrics={}
        key=','.join(str(j['stage']) for j in command)
        entry=self.program_metrics.setdefault(key,dict(programs=0,native_ms=0.,saturated_programs=0,cycles={n:0 for n in COUNTER_NAMES}))
        entry['programs']+=1;entry['native_ms']+=completion.stats.elapsed_ns/1e6
        entry['saturated_programs']+=int(metrics['saturated'])
        for name,value in values.items():entry['cycles'][name]+=value
        return dict(ms=completion.stats.elapsed_ns/1e6,tiles=completion.stats.tiles,error=completion.stats.error,performance=metrics,
                    native_start_ns=completion.start_ns,native_end_ns=completion.end_ns)

    def _run_prepared(self,command):
        with self._program_lock,self._core_lock:
            if command.plan is not self._installed_plan:raise RuntimeError('Stale fixed descriptor plan')
            self._begin_operation();failed=True
            try:
                completion=CoreCompletion()
                code=self.lib.cosmos_fixed_stage(C.byref(command.native),self._fixed_counter_regs,C.byref(completion))
                if code or completion.stats.error:raise RuntimeError('Fixed program failed %d/%d'%(code,completion.stats.error))
                result=self._fixed_result(command,completion);failed=False;return result
            finally:self._end_operation(failed)

    def prepare_projection(self,commands,partials,sums,state,destination,gate,*,first_rows,rows,step,condition,layer,operator,generation):
        if not 1<=len(commands)<=6 or len(partials)!=2 or len(sums)!=2 or len(first_rows)!=len(commands) or len(rows)!=len(commands):raise ValueError('Fixed projection shape')
        if list(first_rows)!=list(range(0,self.assignment.rows,256)) or list(rows)!=[min(256,self.assignment.rows-r) for r in first_rows]:raise ValueError('Fixed projection must cover every local row exactly once')
        if any(c.plan is not self._installed_plan for c in commands):raise RuntimeError('Projection descriptor owner changed')
        count=len(commands);cores=(CoreJob*count)(*(c.native for c in commands));r0=(PairJob*count)();r1=(PairJob*count)()
        epoch=wire_epoch(generation,self.group,step,condition,layer,operator)
        if gate.nbytes<24576:raise ValueError('Padded fixed gate span')
        for i,(first,n) in enumerate(zip(first_rows,rows)):
            if n not in (128,256) or first%256 or first+n>self.assignment.rows:raise ValueError('Fixed projection row window')
            size=n*5120*4;offset=first*5120*4
            command=commands[i].jobs[0]
            if command['c']!=partials[i&1].address or command['rows']!=n or command['flags']!=72:raise ValueError('Projection command and slot binding differ')
            for source,target in ((state.address+offset,destination.address+offset),(partials[i&1].address,destination.address+offset)):
                if source!=target and source<target+size and target<source+size:raise ValueError('Partial TP input/output alias')
            if partials[i&1].address<sums[i&1].address+size and sums[i&1].address<partials[i&1].address+size:raise ValueError('TP sum must have separate committed storage')
            if any(b.nbytes<size for b in (partials[i&1],sums[i&1])) or any(b.nbytes<offset+size for b in (state,destination)):raise ValueError('Fixed projection allocation span')
            for round_id,(src,out) in enumerate(((partials[i&1].address,sums[i&1].address),(sums[i&1].address,destination.address+offset))):
                job=PairJob(self._lane_abi(round_id),src,out,state.address+offset,gate.address,n,5120,first,self.node,(self.rank^(1<<round_id))+1,round_id,round_id,epoch,step,condition_id(condition),layer,operator,40000)
                (r0 if round_id==0 else r1)[i]=job
        compute=(CoreCompletion*count)();reduce=(StreamReduceStats*(count*2))();starts=(C.c_uint64*count)();ends=(C.c_uint64*count)()
        plan=NativeProjection(self.batch_controller.array.ctypes.data,cores,r0,r1,self._batch_bases,self._fixed_counter_regs,compute,reduce,starts,ends,count,120000)
        plan._owners=(cores,r0,r1,compute,reduce,starts,ends,commands,partials,sums,state,destination,gate)
        plan.commands=commands;plan.context=dict(step=step,condition=condition,layer=layer,operator=operator,generation=generation)
        plan.receipt_key=(step,condition,layer,operator)
        with self._state_lock:
            if plan.receipt_key in self._native_projection_receipts:raise RuntimeError('Fixed projection submitted twice')
            self._native_projection_receipts[plan.receipt_key]=dict(status='PREPARED',tiles=count,rows=sum(rows),checked_bytes=0,retired=0)
        return plan

    def _emit_native(self,event):
        sink=getattr(self,'native_event_sink',None)
        if sink is not None:sink(event)

    def compute_projection(self,plan):
        with self._program_lock,self._core_lock:
            if any(c.plan is not self._installed_plan for c in plan.commands):raise RuntimeError('Projection plan replaced while live')
            self._begin_operation();failed=True;owner_start=time.monotonic_ns();owner_cpu=time.thread_time_ns()
            try:
                code=self.lib.cosmos_fixed_project_compute(C.byref(plan))
                for i in range(plan.produced):
                    command=plan.commands[i];completion=plan.compute[i];result=self._fixed_result(command,completion)
                    stages=[j['stage'] for j in command]
                    self._emit_native(dict(category='ffn_down_f16_quantized_on_read' if stages==[16] else 'gemm',stages=stages,descriptors=len(command),
                        start_ns=completion.start_ns,end_ns=completion.end_ns,native_ms=result['ms'],thread=threading.get_ident(),ok=True,
                        native_fixed_plan=True,**plan.context))
                if code:raise RuntimeError('Native fixed projection compute failed %d'%code)
                failed=False;return dict(tiles=plan.produced)
            finally:
                try:
                    self._emit_native(dict(category='fixed_compute_owner',start_ns=owner_start,end_ns=time.monotonic_ns(),thread_cpu_ns=time.thread_time_ns()-owner_cpu,
                        thread=threading.get_ident(),ok=not failed,native_fixed_plan=True,includes_queue_wait=True,**plan.context))
                finally:self._end_operation(failed)

    def reduce_projection(self,plan):
        with self._collective_lock,self._lane_locks[0],self._lane_locks[1]:
            self._begin_operation();failed=True;owner_start=time.monotonic_ns();owner_cpu=time.thread_time_ns()
            try:
                for i in range(plan.count):
                    for round_id,j in enumerate((plan.round0[i],plan.round1[i])):
                        context=(j.epoch,j.step,j.layer,j.operator_id,j.first_row)
                        if context<=self._last_pair_context.get(round_id,(-1,)):raise RuntimeError('Fixed TP plan context replay')
                        self._last_pair_context[round_id]=context
                code=self.lib.cosmos_fixed_project_reduce(C.byref(plan))
                for i in range(plan.retired):
                    self._emit_native(dict(category='tp_reduction',start_ns=plan.tp_start_ns[i],end_ns=plan.tp_end_ns[i],thread=threading.get_ident(),ok=True,native_fixed_plan=True,**plan.context))
                if code:raise RuntimeError('Native fixed projection reduction failed %d'%code)
                if plan.produced!=plan.count or plan.retired!=plan.count or plan.failed or any(plan.slots):raise RuntimeError('Native projection did not retire both committed slots')
                checked=0
                for i in range(plan.count):
                    expected=plan.round0[i].rows*5120*4
                    for round_id in range(2):
                        result=plan.reduce[2*i+round_id]
                        if result.error or result.tx_bytes!=expected or result.rx_bytes!=expected or result.chunks!=plan.round0[i].rows*3:raise RuntimeError('Native projection byte/chunk retirement mismatch')
                        checked+=result.rx_bytes
                with self._state_lock:
                    self._native_projection_receipts[plan.receipt_key].update(status='COMPLETE_TWO_ROUND_DRAIN',checked_bytes=checked,retired=int(plan.retired))
                failed=False;return dict(tiles=plan.retired,drained=True)
            except BaseException as error:
                plan.reduction_error=error
                with self._state_lock:self._native_projection_receipts[plan.receipt_key].update(status='FAILED',error=repr(error),retired=int(plan.retired))
                self._capture_native_failure('fixed_hardware_batch',0,plan.context,code=int(plan.failed) or None,error=error)
                # Wake a producer waiting for a free slot even when admission
                # failed before entering C. The allocation stays poisoned.
                self.lib.cosmos_fixed_project_cancel(C.byref(plan),-15)
                raise
            finally:
                try:
                    self._emit_native(dict(category='tp_native_control_owner',start_ns=owner_start,end_ns=time.monotonic_ns(),thread_cpu_ns=time.thread_time_ns()-owner_cpu,
                        thread=threading.get_ident(),ok=not failed,native_fixed_plan=True,includes_queue_wait=True,**plan.context))
                finally:self._end_operation(failed)

    def native_projection_status(self,step):
        """Receipts from actual successful native returns, never a hardware DONE proxy."""
        with self._state_lock:
            receipts=[value for key,value in self._native_projection_receipts.items() if key[0]==step]
            # Layer0 shares self projection (1), then P/N cross+MLP (4).
            # The remaining35 layers each have P/N self+cross+MLP (6).
            expected=5+35*6;tiles=(self.assignment.rows+255)//256
            return dict(mode=NATIVE_FIXED_TP_CONTROL_MODE,hardware_batch_used=True,step=step,local_rows=self.assignment.rows,expected_plans=expected,
                planned=len(receipts),completed=sum(r['status']=='COMPLETE_TWO_ROUND_DRAIN' for r in receipts),
                expected_tiles=expected*tiles,retired_tiles=sum(r['retired'] for r in receipts),
                expected_checked_bytes=expected*2*self.assignment.rows*5120*4,
                checked_bytes=sum(r['checked_bytes'] for r in receipts),errors=sum(r['status']=='FAILED' for r in receipts),
                completion='FRESH_COMPUTE_DONE_THEN_HARDWARE_BATCH_SUM_FUSED_BYTECOUNT_DRAIN')

    def performance_summary(self):
        return dict(schema='FAST_GROUP_V1',clock_domain='mac_fast_group',target_mhz=300,physical_shards=5,
            programs=getattr(self,'program_metrics',{}),
            clipped_programs_require_host_wall_time=True,
            interpretation='Mutually exclusive whole-group fast-clock occupancy; not five-core sums, effective MAC lane cycles, or measured DDR utilization')
    def performance_snapshot(self):
        """Read only after program completion; the new image keeps eight CSRs.

        All categories partition one five-shard group at 300MHz. LOAD means input
        loading has not completed, not a claim of external DDR starvation.
        """
        mapping=(('elapsed','perf_elapsed_cycles'),('multiply_issue','perf_owner_a_cycles'),
            ('load_a_pending','perf_owner_b_cycles'),('load_b_pending_after_a','perf_source_empty_cycles'),
            ('seed','perf_result_stall_cycles'),('store','perf_a_commits'),
            ('output_blocked','perf_b_commits'),('other','perf_no_request_cycles'))
        # `running` also includes permitted SFP prefetch/TP DMA overlap. Only
        # the core must have completed; serialize with the next descriptor
        # program so its next start cannot reset the CSR bank while reading.
        with self._program_lock:
            self._check_available()
            if not self.core.read(0)&4:raise RuntimeError('Counters require an idle completed core program')
            values={name:int(self.core.read(self.regs[csr])) for name,csr in mapping}
        if sum(values[name] for name,_ in mapping[1:])!=values['elapsed']:
            raise RuntimeError('FAST_GROUP_V1 interval partition does not sum')
        return dict(schema='FAST_GROUP_V1',clock_domain='mac_fast_group',target_mhz=300,physical_shards=5,
            cycles=values,saturated=values['elapsed']==0xffffffff,
            clipped=values['elapsed']==0xffffffff,maximum_counter_interval_seconds=0xffffffff/300000000,
            snapshot_scope='LAST_COMPLETED_PROGRAM',not_dsp_utilization_percent=True)
    def port_status(self,required_mask=15):
        if required_mask&~15:raise ValueError('Unknown port')
        rows=[];up=0
        for lane in self.interfaces:
            status=lane['link'].read(4);errors=sum(lane['link'].read(o) for o in (0x4c,0x50,0xd4))
            if status&32:up|=1<<lane['lane']
            rows.append(dict(port=lane['lane'],up=bool(status&32),errors=errors))
        return dict(required_mask=required_mask,up_mask=up,ready=(up&required_mask)==required_mask and not any(x['errors'] for x in rows if required_mask&(1<<x['port'])),ports=rows)
    def snapshot(self):
        value=C206Board.snapshot(self)
        value.update(hardware_version=PROFILE['kind'],hardware_id=PROFILE['hardware_id'],runtime_id=PROFILE['runtime_id'],
            compute_revision='TP4_SP3_COMMON_IP'+PROFILE['ip_version'],interface_count=4,stage=self.group,rank=self.rank,
            profile_sha256=PROFILE['profile_sha256'],reference_fclk0_mhz_readback=self.clock,
            target_control_mhz=PROFILE['slow_mhz'],target_mac_mhz=PROFILE['fast_mhz'],
            actual_control_and_mac_frequency_measured=False,ports=self.port_status(15))
        return value
    def wait_link(self,required_mask=15,timeout=40):
        end=time.monotonic()+timeout
        while not self.port_status(required_mask)['ready']:
            if time.monotonic()>end:raise TimeoutError('Required SFP ports unavailable')
            time.sleep(.01)
    def initialize_startup(self,required_mask,timeout=40,*,cancel_check=None):
        """Record and clear training-period counters once, before cohort START."""
        if getattr(self,'_startup_clear_done',False):raise RuntimeError('Startup counter clear is one-use, never a runtime error waiver')
        if cancel_check is not None:cancel_check()
        end=time.monotonic()+timeout
        selected=[x for x in self.interfaces if required_mask&(1<<x['lane'])]
        stable_since=None;previous_faults=None
        while True:
            if cancel_check is not None:cancel_check()
            up=all(l['link'].read(4)&32 for l in selected)
            faults=tuple(int(l['link'].read(0x50)) for l in selected)
            current=time.monotonic()
            if up and faults==previous_faults:
                if stable_since is None:stable_since=current
                if current-stable_since>=.5:break
            else:stable_since=None
            previous_faults=faults
            if current>end:raise TimeoutError('Required PHYs/MAC fault counters did not settle before the one startup clear')
            time.sleep(.01)
        with self._state_lock:
            if self._inflight or self.running or not self.core.read(0)&4:raise RuntimeError('Startup clear requires idle compute and no issued operations')
            before=self.port_status(required_mask)
            self._startup_clear_record=dict(before_clear=before,attach_counter_history=getattr(self,'_attach_counter_history',[]),pre_clear_stable_seconds=.5,
                stable_mac_fault_counts=list(previous_faults))
            for lane in selected:
                for dma,offsets in [(lane['dma'],(0,0x30))]+([(lane['residual_dma'],(0,))] if lane['residual_dma'] is not None else []):
                    for off in offsets:
                        if dma.read(off+4)&0x70 or dma.read(off+0x28):raise RuntimeError('Startup DMA had errors or submitted bytes')
                lane['link'].write(4,2);lane['reducer'].write(8,2)
            self._startup_clear_done=True
        clean_since=None
        while True:
            if cancel_check is not None:cancel_check()
            after=self.port_status(required_mask)
            self._startup_clear_record['after_clear']=after
            quiet=all(not(l['link'].read(4)&0x5f) and not l['link'].read(0x80) and not l['link'].read(0xc0)
                and not l['link'].read(0xc4) and not l['link'].read(0xe0) and not(l['reducer'].read(4)&0x15) for l in selected)
            if any(row['errors'] for row in after['ports'] if required_mask&(1<<row['port'])):raise RuntimeError('Faults resumed after the one startup clear; no additional clear is allowed')
            if after['ready'] and quiet:
                if clean_since is None:clean_since=time.monotonic()
                if time.monotonic()-clean_since>=.5:return dict(ready=True,attach_counter_history=getattr(self,'_attach_counter_history',[]),before_clear=before,after_clear=after,
                    pre_clear_stable_seconds=.5,post_clear_clean_seconds=.5,one_startup_clear=True,runtime_error_clear=False)
            else:clean_since=None
            if time.monotonic()>end:raise RuntimeError('Required fabric not clean after its one recorded startup CLEAR')
            time.sleep(.001)
    def _lane_abi(self,i):
        l=self.interfaces[i]
        return StreamLane(*(int(l[n].array.ctypes.data) if l[n] is not None else 0 for n in ('dma','residual_dma','link','reducer')))
    def _capture_native_failure(self,operation,lane_id,context,*,code=None,stats=None,expected_bytes=None,error=None):
        """Failure-only reads before DMA reset; diagnostics never replace the error."""
        try:
            record=dict(operation=operation,stage=self.group,rank=self.rank,node=self.node,
                physical_node=self.group*4+self.node,lane=lane_id,context=context,
                code=code,error=repr(error) if error is not None else None,
                expected_bytes=expected_bytes,captured_monotonic_ns=time.monotonic_ns(),
                capture_phase='WAIT_LINK_FAILURE_BEFORE_NATIVE' if operation.endswith('_wait_link') else 'NATIVE_RETURN_BEFORE_DMA_RESET',
                csr_read_consistency='SEQUENTIAL_VOLATILE_READS',csr={},read_errors={})
            if stats is not None:record['native_stats']={name:int(getattr(stats,name)) for name,_ in stats._fields_}
            lane=self.interfaces[lane_id]
            offsets={
                'link':(0x04,0x08,0x0c,0x30,0x34,0x40,0x4c,0x50,0x80,0x90,0x94,
                        0x9c,0xa0,0xa4,0xa8,0xac,0xb0,0xb4,0xb8,0xc0,0xc4,0xc8,0xcc,0xd0,0xd4,0xd8,0xdc,0xe0),
                'dma':(0x00,0x04,0x18,0x1c,0x28,0x30,0x34,0x48,0x4c,0x58),
                'residual_dma':(0x00,0x04,0x18,0x1c,0x28),
                'reducer':(0x04,0x0c,0x10,0x14,0x18,0x1c,0x20,0x24,0x28)}
            for device,addresses in offsets.items():
                registers=lane.get(device)
                if registers is None:record['csr'][device]=None;continue
                values=record['csr'][device]={}
                for offset in addresses:
                    key=f'0x{offset:02x}'
                    try:values[key]=int(registers.read(offset))
                    except BaseException as read_error:record['read_errors'][device+':'+key]=repr(read_error)
            # Native calls on other lanes can fail while peers are being cancelled.
            # Preserve the earlier failure as well as the most recent snapshot.
            with self._state_lock:
                history=getattr(self,'native_failure_history',[])
                history.append(record);self.native_failure_history=history[-8:]
                self.last_native_failure=record
        except BaseException as capture_error:
            try:self.native_failure_capture_error=repr(capture_error)
            except BaseException:pass
    def allreduce_tile(self,partial,summed,state,gate,*,first_row,step,condition,layer,operator,generation,rows=256,output_state=None):
        p=PROFILE;n=rows*p['hidden']*4
        destination=state if output_state is None else output_state
        if rows not in p['partial_rows'] or first_row<0 or first_row%128 or first_row+rows>self.assignment.rows:
            raise ValueError('SP3 partial tile rows or local extent')
        if any(b.nbytes<n for b in (partial,summed,state,destination)) or gate.nbytes<((p['hidden']*4+8191)//8192)*8192:raise ValueError('TP4 tile/padded gate capacity')
        if destination.address!=state.address and destination.address<state.address+n and state.address<destination.address+n:
            raise ValueError('Out-of-place TP4 residual output must not partially alias its input')
        if destination.address!=partial.address and destination.address<partial.address+n and partial.address<destination.address+n:
            raise ValueError('Chunk pipeline output must not partially overlap a future SUM input')
        epoch=wire_epoch(generation,self.group,step,condition,layer,operator);stats=[]
        with self._collective_lock:
            marker=self._begin_operation();failed=True
            try:
                # Each complete SUM row segment can enter FUSED while the
                # next segment is still being reduced on the other lane.
                # The existing intermediate tile owns the committed data;
                # no CPU tensor copy, new thread, or extra PL buffer is used.
                with self._lane_locks[0],self._lane_locks[1]:
                    jobs=[]
                    for round_id,(src,out) in enumerate(((partial,summed),(summed,destination))):
                        context=(epoch,step,layer,operator,first_row)
                        if context<=self._last_pair_context.get(round_id,(-1,)):raise RuntimeError('TP4 context must advance')
                        self._last_pair_context[round_id]=context
                        j=PairJob(self._lane_abi(round_id),src.address,out.address,state.address,gate.address,rows,p['hidden'],first_row,self.node,(self.rank^(1<<round_id))+1,round_id,round_id,epoch,step,condition_id(condition),layer,operator,40000)
                        jobs.append(j)
                    results=(StreamReduceStats*2)()
                    if self.batch_controller.read(4)&1:raise RuntimeError('Previous hardware batch still owns DMA registers')
                    code=self.lib.cosmos_tp4_pair_batch(self.batch_controller.array.ctypes.data,C.byref(jobs[0]),C.byref(jobs[1]),self._batch_bases,results)
                    for round_id,s in enumerate(results):
                        if code or s.error or s.tx_bytes!=n or s.rx_bytes!=n or s.chunks!=rows*3:
                            self._capture_native_failure('allreduce_hardware_batch',round_id,
                                dict(generation=generation,epoch=epoch,step=step,condition=condition,
                                     layer=layer,operator=operator,first_row=first_row,rows=rows,
                                     cols=p['hidden'],round=round_id,peer=jobs[round_id].peer,timeout_ms=jobs[round_id].timeout_ms),
                                code=code,stats=s,expected_bytes=n)
                            raise RuntimeError(f'TP4 hardware batch round {round_id}: code={code} error={s.error}')
                        stats.append(dict(round=round_id,ms=s.elapsed_ns/1e6,bytes=s.rx_bytes,chunks=s.chunks))
                failed=False;return dict(rounds=stats,rounds_overlap=True,packet_credit_per_lane=1,
                    schedule='TP4_FP32_HARDWARE_BATCH_V1',cpu_tensor_arithmetic=False,gate_residual_applications=1)
            finally:self._end_operation(failed)
    def raw_exchange(self,lane_id,peer,tx,rx,nbytes,*,epoch,step,channel,sequence,flush=True,invalidate=True,timeout_ms=43200000):
        if not 0<=lane_id<4 or nbytes>8192 or nbytes%8 or tx.nbytes<nbytes or rx.nbytes<nbytes:raise ValueError('Raw DMA bounds')
        node=self.node if lane_id<2 else self.group+1
        expected=((self.rank^(1<<lane_id))+1) if lane_id<2 else ((self.group+(1 if lane_id==2 else 2))%3+1)
        if peer!=expected:raise ValueError('Peer is not the direct physical SP3 neighbor')
        context=(epoch,step,channel,sequence)
        with self._lane_locks[lane_id]:
            key=(lane_id,channel)
            if context<=self._last_bulk_context.get(key,(-1,)):raise RuntimeError('Raw context replay')
            self._last_bulk_context[key]=context
            try:self.wait_link(1<<lane_id)
            except BaseException as error:
                self._capture_native_failure('raw_exchange_wait_link',lane_id,
                    dict(epoch=epoch,step=step,channel=channel,sequence=sequence,peer=peer),
                    expected_bytes=nbytes,error=error)
                raise
            marker=self._begin_operation();failed=True
            try:
                if flush:tx.root.flush()
                j=BulkJob(self._lane_abi(lane_id),tx.address,rx.address,nbytes,node,peer,lane_id,epoch,step,channel,sequence,timeout_ms)
                s=StreamReduceStats();code=self.lib.cosmos_tp4_bulk(C.byref(j),C.byref(s))
                if code or s.error or s.rx_bytes!=nbytes:
                    self._capture_native_failure('raw_exchange',lane_id,
                        dict(epoch=epoch,step=step,channel=channel,sequence=sequence,peer=peer,timeout_ms=timeout_ms),
                        code=code,stats=s,expected_bytes=nbytes)
                    raise RuntimeError(f'Raw DMA failed {code}/{s.error}')
                if invalidate:rx.root.invalidate()
                failed=False;return dict(ms=s.elapsed_ns/1e6,bytes=s.rx_bytes)
            finally:self._end_operation(failed)
    def raw_span(self,lane_id,peer,tensor,*,sending,epoch,step,channel,timeout_ms=43200000):
        if tensor.nbytes%8 or tensor.nbytes>512*1024**2:raise ValueError('Span bounds')
        node=self.node if lane_id<2 else self.group+1
        expected=((self.rank^(1<<lane_id))+1) if lane_id<2 else ((self.group+(1 if lane_id==2 else 2))%3+1)
        if peer!=expected:raise ValueError('Peer is not the direct physical SP3 neighbor')
        context=(epoch,step,channel,0)
        with self._lane_locks[lane_id]:
            key=(lane_id,channel)
            if context<=self._last_bulk_context.get(key,(-1,)):raise RuntimeError('Span replay')
            self._last_bulk_context[key]=context
            try:self.wait_link(1<<lane_id)
            except BaseException as error:
                self._capture_native_failure('raw_span_wait_link',lane_id,
                    dict(epoch=epoch,step=step,channel=channel,peer=peer,sending=bool(sending)),
                    expected_bytes=tensor.nbytes,error=error)
                raise
            marker=self._begin_operation();failed=True
            try:
                tx,rx=self._bulk_buffers[lane_id]
                j=SpanJob(self._lane_abi(lane_id),tensor.address,tx.address,rx.address,tensor.nbytes,node,peer,lane_id,epoch,step,channel,timeout_ms,int(sending))
                s=StreamReduceStats();code=self.lib.cosmos_tp4_bulk_span(C.byref(j),C.byref(s))
                if code or s.error or s.rx_bytes!=tensor.nbytes or s.tx_bytes!=tensor.nbytes:
                    self._capture_native_failure('raw_span',lane_id,
                        dict(epoch=epoch,step=step,channel=channel,peer=peer,sending=bool(sending),timeout_ms=timeout_ms),
                        code=code,stats=s,expected_bytes=tensor.nbytes)
                    raise RuntimeError(f'Native span failed {code}/{s.error}')
                failed=False;return dict(ms=s.elapsed_ns/1e6,bytes=s.rx_bytes,native_chunks=s.chunks,host_submissions=1)
            finally:self._end_operation(failed)
    def raw_duplex_span(self,lane_id,peer,tx,rx,*,epoch,step,channel,timeout_ms=43200000):
        if lane_id not in (2,3) or any(v.nbytes<=0 or v.nbytes%64 or v.nbytes>512*1024**2 or v.address%64 for v in (tx,rx)):
            raise ValueError('KV duplex span requires aligned nonempty direct-neighbor buffers')
        if tx.address<rx.address+rx.nbytes and rx.address<tx.address+tx.nbytes:raise ValueError('KV send and receive spans overlap')
        node=self.group+1;expected=(self.group+(1 if lane_id==2 else 2))%3+1
        if peer!=expected:raise ValueError('KV duplex peer is not physical neighbor')
        context=(epoch,step,channel,0)
        with self._lane_locks[lane_id]:
            key=(lane_id,channel)
            if context<=self._last_bulk_context.get(key,(-1,)):raise RuntimeError('KV duplex replay')
            self._last_bulk_context[key]=context
            self.wait_link(1<<lane_id);marker=self._begin_operation();failed=True
            try:
                dummy_tx,dummy_rx=self._bulk_buffers[lane_id]
                job=DuplexSpanJob(self._lane_abi(lane_id),tx.address,rx.address,dummy_tx.address,dummy_rx.address,
                    tx.nbytes,rx.nbytes,node,peer,lane_id,epoch,step,channel,timeout_ms)
                stats=StreamReduceStats();code=self.lib.cosmos_tp4_bulk_duplex_span(C.byref(job),C.byref(stats))
                wire=max(tx.nbytes,rx.nbytes)
                if code or stats.error or stats.tx_bytes!=wire or stats.rx_bytes!=wire:
                    self._capture_native_failure('raw_duplex_span',lane_id,dict(epoch=epoch,step=step,channel=channel,peer=peer),code=code,stats=stats,expected_bytes=wire)
                    raise RuntimeError(f'KV duplex failed {code}/{stats.error}')
                failed=False
                return dict(ms=stats.elapsed_ns/1e6,bytes=wire,useful_tx_bytes=tx.nbytes,useful_rx_bytes=rx.nbytes,native_chunks=stats.chunks,host_submissions=1)
            finally:self._end_operation(failed)
    def batch_status(self):
        status=int(self.batch_controller.read(4))
        return dict(status=status,busy=bool(status&1),idle=not bool(status&1),error=int(self.batch_controller.read(12)) if status&4 else 0,completed=bool(status&2),drained=bool(status&8),sum_retired=int(self.batch_controller.read(0x38)),fused_retired=int(self.batch_controller.read(0x3c)),phase=int(self.batch_controller.read(0x90)))
    def cancel_native(self):
        self.lib.cosmos_tp4_cancel(1)
        if hasattr(self,'batch_controller') and self.batch_controller.read(4)&1:self.batch_controller.write(8,4)
    def stop_owned_dma(self):
        """After native workers join, reset only this leased board's DMA masters."""
        if getattr(self,'_hardware_lease',None) is None or self._hardware_lease.closed:raise RuntimeError('No owned hardware lease; refuse DMA access')
        if self._inflight:raise RuntimeError('Native workers must join before DMA reset')
        if self.batch_controller.read(4)&1:raise RuntimeError('Batch AXI-Lite master has not retired; DMA reset would race its writes')
        if not getattr(self,'_failed_dma_reset',False):
            for lane in self.interfaces:
                for dma in (lane['dma'],lane['residual_dma']):
                    if dma is None:continue
                    dma.write(0,4);end=time.monotonic()+5
                    while dma.read(0)&4:
                        if time.monotonic()>end:raise TimeoutError('Owned failure reset did not drain')
            self._failed_dma_reset=True
        if not self.core.read(0)&4:return False
        for lane in self.interfaces:
            for dma,offsets in [(lane['dma'],(0,0x30))]+([(lane['residual_dma'],(0,))] if lane['residual_dma'] is not None else []):
                if any(not(dma.read(off+4)&1) or dma.read(off+4)&0x70 for off in offsets):return False
        self.poisoned=False;self.running=False;self.needs_reinitialization=True;return True
    def close(self):
        with self._state_lock:
            if self._inflight or self.poisoned or self.batch_controller.read(4)&1:raise RuntimeError('Failed/in-flight native operations retain DMA allocations')
            for lane in self.interfaces:
                for dma,offsets in [(lane['dma'],(0,0x30))]+([(lane['residual_dma'],(0,))] if lane['residual_dma'] is not None else []):
                    for off in offsets:
                        status=dma.read(off+4)
                        if status&0x70 or not(status&3) and dma.read(off+0x28):raise RuntimeError('DMA not drained')
                        dma.write(off,0)
            for b in self.buffers:b.check()
            while self.buffers:
                self.buffers[0].close();self.buffers.pop(0)
            self.running=False;self._hardware_lease.close();self._hardware_lease=None
