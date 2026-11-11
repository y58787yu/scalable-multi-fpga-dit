"""Explicit finite two/four-role file permits; no SSH, scheduling or background launch."""
import json
import re
import time
import uuid
from pathlib import Path
from tp4_profile import PROFILE,PRESENT_PORTS,ACTIVE_PORTS
from tp4_startup import atomic_json

ROLES={(g,r) for g in range(PROFILE['sequence_parallel']) for r in range(2)}
COMMUNICATION_PROTOCOL='IP20_TP2_P1_FUSED_SP_P2_SERIAL_FULL_DUPLEX_V1'
NATIVE_CHUNK_PIPELINE_REVISION=1
NATIVE_BATCH_REVISION=2
NATIVE_FIXED_TP_CONTROL_MODE=5
COMMUNICATION_BINDINGS=('communication_protocol','native_chunk_pipeline_revision','native_batch_revision','native_fixed_tp_control_mode')
BINDINGS=('cohort','generation','profile_sha256','source_binding_sha256','source_manifest_sha256','schedule_sha256','updates','seed','execution_steps')+COMMUNICATION_BINDINGS

def validate_communication_contract(value):
    if (not isinstance(value,dict) or value.get('communication_protocol')!=COMMUNICATION_PROTOCOL
        or type(value.get('native_chunk_pipeline_revision'))is not int
        or value['native_chunk_pipeline_revision']!=NATIVE_CHUNK_PIPELINE_REVISION
        or type(value.get('native_batch_revision'))is not int or value['native_batch_revision']!=NATIVE_BATCH_REVISION
        or type(value.get('native_fixed_tp_control_mode'))is not int or value['native_fixed_tp_control_mode']!=NATIVE_FIXED_TP_CONTROL_MODE):
        raise ValueError('All SP3 roles require the IP16 corrected hardware TP batch control; mixed runtime rejected')
    return value

def communication_binding(board):
    """Read the loaded library's pure identity function; no MMIO or wire work."""
    value={key:getattr(board,key,None) for key in COMMUNICATION_BINDINGS}
    validate_communication_contract(value)
    probe=getattr(getattr(board,'lib',None),'cosmos_tp4_chunk_pipeline_revision',None)
    if not callable(probe):raise RuntimeError('Loaded native executor lacks the TP4 chunk protocol identity')
    actual=probe()
    if type(actual)is not int or actual!=NATIVE_CHUNK_PIPELINE_REVISION:
        raise RuntimeError('Loaded native executor differs from the verified TP4 chunk protocol')
    batch_probe=getattr(getattr(board,'lib',None),'cosmos_tp4_batch_revision',None)
    if not callable(batch_probe) or batch_probe()!=NATIVE_BATCH_REVISION:
        raise RuntimeError('Loaded native executor does not implement the fixed TP batch controller')
    mode_probe=getattr(getattr(board,'lib',None),'cosmos_fixed_tp_control_mode',None)
    if not callable(mode_probe) or mode_probe()!=NATIVE_FIXED_TP_CONTROL_MODE:
        raise RuntimeError('Loaded native executor differs from the corrected hardware TP batch control mode')
    return value

def validate_quiescence(proof):
    if not isinstance(proof,dict) or proof.get('proven') is not True:raise ValueError('Missing actual quiescence proof')
    zeros=('native_inflight','tile_readers','weight_readers','network_prefetch_pending','disk_prefetch_inflight','pending_async_tasks')
    if any(proof.get(k)!=0 for k in zeros) or proof.get('weight_active') is not False or proof.get('poisoned') is not False or proof.get('running') is not False:raise ValueError('Live owner at step boundary')
    if proof.get('dma_reset') is not False or proof.get('cma_released') is not False or proof.get('next_step_submitted') is not False:raise ValueError('Invalid quiescence method')
    core=proof.get('core_control',0)
    if type(core)is not int or not core&4 or core&0x81:raise ValueError('Compute is not idle')
    batch=proof.get('tp_batch',{})
    if (batch.get('idle') is not True or batch.get('busy') is not False or batch.get('error')!=0
        or batch.get('completed') is not True or batch.get('drained') is not True):
        raise ValueError('Hardware TP batch has not completed and drained its final tile')
    native=proof.get('tp_native_pipeline',{});rows=native.get('local_rows')
    if (native.get('mode')!=NATIVE_FIXED_TP_CONTROL_MODE or native.get('hardware_batch_used') is not True
        or type(rows)is not int or rows not in (2048,4096) or native.get('errors')!=0
        or type(native.get('step'))is not int or not 0<=native['step']<19
        or native.get('expected_plans')!=167 or native.get('planned')!=167 or native.get('completed')!=167
        or native.get('expected_tiles')!=167*((rows+255)//256) or native.get('retired_tiles')!=167*((rows+255)//256)
        or native.get('expected_checked_bytes')!=167*rows*2048*4 or native.get('checked_bytes')!=167*rows*2048*4
        or native.get('completion')!='FRESH_COMPUTE_DONE_THEN_TP2_FUSED_BYTECOUNT_DRAIN'):
        raise ValueError('Native SUM/FUSED projections lack complete real byte-count retirement')
    # Both finite profiles end with 256 rows, one 8192-byte FUSED packet per row.
    final_chunks=rows%256 or 256
    if batch.get('sum_retired')!=0 or batch.get('fused_retired')!=final_chunks:
        raise ValueError('Final hardware batch SUM/FUSED packet counts differ from the local tail')
    lanes=proof.get('lanes',[])
    if len(lanes)!=len(PRESENT_PORTS) or {x.get('port') for x in lanes}!=set(PRESENT_PORTS):raise ValueError('Both present P1/P2 lane proofs required')
    for lane in lanes:
        status=lane.get('link_status',0);reducer=lane.get('reducer_status',0)
        active=lane['port'] in ACTIVE_PORTS
        errors=lane.get('errors');pending=lane.get('pending');channels=lane.get('dma_status')
        if active and (type(status)is not int or status&0x59 or not status&32 or type(reducer)is not int or reducer&0x15 or lane.get('reducer_errors')!=0):raise ValueError('Link/reducer not drained')
        if active and (not isinstance(errors,list) or len(errors)!=3 or any(type(x)is not int or x for x in errors)):raise ValueError('Link errors')
        if active and (not isinstance(pending,list) or len(pending)!=4 or any(type(x)is not int or x for x in pending)):raise ValueError('Link pending work')
        if not isinstance(channels,list) or len(channels) not in (2,3):raise ValueError('DMA channels missing')
        for channel in channels:
            value=channel.get('status');length=channel.get('last_submitted_bytes')
            if type(value)is not int or type(length)is not int or length<0 or value&0x70 or (length and not value&3):raise ValueError('DMA owns submitted data or has errors')
    return proof

def quiescence(board,model,*,timeout=5,cancel_check=None):
    """Only reads actual ownership and CSR state; no CLEAR/reset/DDR release."""
    for future in list(model.weights.pending.values()):
        while not future.done():
            if cancel_check is not None:cancel_check()
            time.sleep(.01)
        future.result()
    end=time.monotonic()+timeout
    while True:
        if cancel_check is not None:cancel_check()
        with board._state_lock:
            ownership=model.asynchronous_ownership()
            live=bool(board._inflight or model.tiles.busy or ownership['pending_async_tasks'] or model.weights.active is not None or model.weights.readers or model.weights.network_pending)
            if live:raise RuntimeError('SP3 step reached with live native/tile/weight/network ownership')
            core=int(board.core.read(0));lanes=[]
            for lane in board.interfaces:
                link=lane['link'];reducer=lane['reducer'];channels=[]
                for dma,offsets in [(lane['dma'],(0,0x30))]+([(lane['residual_dma'],(0,))] if lane['residual_dma'] is not None else []):
                    for off in offsets:channels.append(dict(status=int(dma.read(off+4)),last_submitted_bytes=int(dma.read(off+0x28))))
                lanes.append(dict(port=lane['lane'],link_status=int(link.read(4)),reducer_status=int(reducer.read(4)),reducer_errors=int(reducer.read(0x18)),errors=[int(link.read(o)) for o in (0x4c,0x50,0xd4)],pending=[int(link.read(o)) for o in (0x80,0xc0,0xc4,0xe0)],dma_status=channels))
            proof=dict(proven=True,core_control=core,lanes=lanes,tp_batch=board.batch_status(),tp_native_pipeline=board.native_projection_status(model.step),native_inflight=int(board._inflight),tile_readers=int(bool(model.tiles.busy)),pending_async_tasks=ownership['pending_async_tasks'],weight_readers=len(model.weights.readers),weight_active=model.weights.active is not None,network_prefetch_pending=len(model.weights.network_pending),disk_prefetch_inflight=sum(not f.done() for f in model.weights.pending.values()),poisoned=bool(board.poisoned),running=bool(board.running),next_step_submitted=False,dma_reset=False,cma_released=False)
        try:return validate_quiescence(proof)
        except ValueError as error:
            if time.monotonic()>end:raise RuntimeError('SP3 step did not drain: '+json.dumps(proof)) from error
        time.sleep(.01)

def validate_checkpoint(checkpoint,receipt):
    if not isinstance(checkpoint,dict) or checkpoint.get('status')!='SP3_STEP_CHECKPOINT_SAVED':raise ValueError('Root checkpoint absent')
    if any(checkpoint.get(k)!=receipt.get(k) for k in BINDINGS) or checkpoint.get('group')!=0 or checkpoint.get('rank')!=0 or checkpoint.get('completed_step')!=receipt.get('step') or checkpoint.get('next_step')!=receipt.get('step')+1:raise ValueError('Root checkpoint binding mismatch')
    if checkpoint.get('shape')!=[1,16,1,128,128] or checkpoint.get('dtype')!='float32' or checkpoint.get('latent_bytes')!=1048576 or not checkpoint.get('path'):raise ValueError('Checkpoint tensor contract')
    if any(not isinstance(checkpoint.get(k),str) or not re.fullmatch('[0-9a-f]{64}',checkpoint[k]) for k in ('latent_sha256','file_sha256')):raise ValueError('Checkpoint actual byte digests absent')
    return checkpoint

def permit(documents,phase):
    documents=list(documents)
    if phase not in ('INITIALIZE','RELEASE','RESUME') or len(documents)!=PROFILE['boards']:raise ValueError('Exactly the selected two/four TP2 role receipts required')
    state={'INITIALIZE':'PREPARED','RELEASE':'READY','RESUME':'STEP_DONE'}[phase]
    first=documents[0];seen={}
    for doc in documents:
        validate_communication_contract(doc)
        role=(doc.get('group'),doc.get('rank'))
        if doc.get('status')!='SP3_'+state or role not in ROLES or role in seen or not doc.get('token') or any(doc.get(k)!=first.get(k) for k in BINDINGS):raise ValueError('SP3 permit identity/readiness')
        if phase=='RESUME':
            if doc.get('step')!=first.get('step') or not doc.get('ready_token'):raise ValueError('Step barrier mismatch')
            validate_quiescence(doc.get('quiescence'))
            if doc['quiescence']['tp_native_pipeline']['step']!=doc['step']:raise ValueError('Native projection proof belongs to another step')
        seen[role]=doc
    result={k:first[k] for k in BINDINGS};result.update(status='SP3_'+phase,roles=[dict(group=g,rank=r,token=seen[g,r]['token'],**{key:seen[g,r][key] for key in COMMUNICATION_BINDINGS}) for g,r in sorted(ROLES)])
    if phase=='RESUME':
        result['step']=first['step'];result['checkpoint']=validate_checkpoint(seen[0,0].get('checkpoint'),first)
        result['roles']=[dict(group=g,rank=r,token=seen[g,r]['token'],ready_token=seen[g,r]['ready_token'],quiescence=seen[g,r]['quiescence'],**{key:seen[g,r][key] for key in COMMUNICATION_BINDINGS}) for g,r in sorted(ROLES)]
    return result

def validate_permit(value,receipt,phase):
    validate_communication_contract(receipt)
    validate_communication_contract(value)
    if value.get('status')!='SP3_'+phase or any(value.get(k)!=receipt.get(k) for k in BINDINGS):raise ValueError('SP3 permit belongs to another source/run')
    roles=value.get('roles',[]);mapping={(x.get('group'),x.get('rank')):x.get('token') for x in roles}
    if len(roles)!=PROFILE['boards'] or set(mapping)!=ROLES or not all(mapping.values()) or mapping[receipt['group'],receipt['rank']]!=receipt['token']:raise ValueError('Missing or stale role token')
    for item in roles:validate_communication_contract(item)
    if phase=='RESUME':
        if value.get('step')!=receipt.get('step'):raise ValueError('Stale resume step')
        for item in roles:
            if not item.get('ready_token'):raise ValueError('Resume lacks attach token')
            validate_quiescence(item.get('quiescence'))
            if item['quiescence']['tp_native_pipeline']['step']!=receipt['step']:raise ValueError('Native projection proof belongs to another step')
        own=next(x for x in roles if (x['group'],x['rank'])==(receipt['group'],receipt['rank']))
        if own['ready_token']!=receipt.get('ready_token') or own['quiescence']!=receipt.get('quiescence'):raise ValueError('Resume replaced actual local proof')
        checkpoint=validate_checkpoint(value.get('checkpoint'),receipt)
        if receipt['group']==0 and receipt['rank']==0 and checkpoint!=receipt.get('checkpoint'):raise ValueError('Resume replaced root checkpoint')

def await_permit(path,receipt,phase,*,timeout,cancel_check=None):
    end=time.monotonic()+timeout;path=Path(path)
    while True:
        if cancel_check is not None:cancel_check()
        if path.is_file():
            value=json.loads(path.read_text(encoding='utf-8'));validate_permit(value,receipt,phase);return value
        if time.monotonic()>end:raise TimeoutError('Missing explicit SP3 '+phase+' permit')
        time.sleep(.05)

def startup(board,role,binding,folder,*,timeout=43200,cancel_check=None):
    validate_communication_contract(binding)
    if any(binding[key]!=value for key,value in communication_binding(board).items()):
        raise RuntimeError('Startup binding differs from the loaded communication implementation')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    base=dict(binding,group=role.group,rank=role.rank)
    prepared=dict(base,status='SP3_PREPARED',token=uuid.uuid4().hex)
    atomic_json(folder/'prepared.json',prepared)
    await_permit(folder/'initialize.json',prepared,'INITIALIZE',timeout=timeout,cancel_check=cancel_check)
    initialization=board.initialize_startup(sum(1<<port for port in ACTIVE_PORTS),timeout=40,cancel_check=cancel_check)
    if not initialization.get('ready'):raise RuntimeError('Selected profile SFP ports are not ready')
    ready=dict(base,status='SP3_READY',token=uuid.uuid4().hex,initialization=initialization)
    atomic_json(folder/'ready.json',ready)
    await_permit(folder/'release.json',ready,'RELEASE',timeout=timeout,cancel_check=cancel_check)
    return ready

def step_gate(ready,step,folder,*,board,model,checkpoint=None,timeout=43200,cancel_check=None):
    proof=quiescence(board,model,cancel_check=cancel_check)
    folder=Path(folder);receipt=dict(ready,status='SP3_STEP_DONE',ready_token=ready['token'],token=uuid.uuid4().hex,step=step,checkpoint=checkpoint,quiescence=proof)
    if ready['group']==0 and ready['rank']==0:validate_checkpoint(checkpoint,receipt)
    atomic_json(folder/f'step_{step:02d}_done.json',receipt)
    await_permit(folder/f'step_{step:02d}_resume.json',receipt,'RESUME',timeout=timeout,cancel_check=cancel_check)
    return receipt
