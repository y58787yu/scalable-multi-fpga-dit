"""One compute owner, one ordered TP worker and one ordered SP worker.

CFG branches have independent DDR scratch. Only a completed GEMM program
publishes its tile; no TP worker reads mutable scheduler fields or live output.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import threading
import time
import numpy as np
from model_runner import View
from model_runner_tp4 import TP4StageModel
from c209_fixed_weights import load_compact,weight_views
from fixed_native_plan import LayerProgramPlan
from c209_tensor_layout import PackedKVLayout
from c209_typed_job import bind_job, PackedKVBinding
from sp3_layout import tiles, CONDITIONS
from sp3_memory import memory_plan
from sp3_assets import SharedWeightStore
from sp3_weight_file_prefetch import OwnedWeightFilePrefetch
from five_fixed_jobs import FixedJobs
from sp3_overlap_schedule import (BranchContext, BranchSlotPools,
    ProjectionCompletion, SCRATCH_NAMES, SCHEDULE)


class SP3Model(TP4StageModel):
    def __init__(self, board, assets, transport, *, generation, budget_bytes=880*1024**2,
                 cancel_check=None):
        self.weights = self.pool = self.sp_pool = None
        self._futures = []
        try:
            self._initialize(board, assets, transport, generation=generation,
                budget_bytes=budget_bytes, cancel_check=cancel_check)
        except BaseException:
            self.join_workers()
            raise

    def _initialize(self, board, assets, transport, *, generation, budget_bytes,
                    cancel_check=None):
        self.b, self.assets, self.role, self.transport = board, assets, assets.role, transport
        self.generation = generation
        self._compute_thread = threading.get_ident()
        n = self.role.rows
        self.p = dict(tokens=n, hidden=5120, local_width=1280, local_heads=10,
                      local_ffn=5120, tile_rows=256, logical_cross_keys=512)
        self.plan = memory_plan(self.role, [x['bytes'] for _, x in assets.layers],
            [x['bytes'] for x in assets.cache_entries], assets.global_layout['bytes'],
            budget_bytes=budget_bytes, updates=assets.updates)
        self.weights = SharedWeightStore(board, assets, transport)
        self.tensors = {name: board.make(shape, dtype) for name, (shape, dtype) in self.plan['shapes'].items()}
        self.fixed_jobs = FixedJobs(self.role)
        with np.load(assets.path(assets.manifest['rope']), allow_pickle=False) as rope:
            for name in ('cos', 'sin'):
                value = rope[name]
                if value.shape != (4096, 128) or value.dtype != np.dtype('<f4'):
                    raise ValueError('Global RoPE source')
                target = self.tensors[name]
                target.array[:] = value[self.role.token_offset:self.role.token_offset+n]
                target.root.flush()
        self.global_parent, self.globals = load_compact(board, assets.global_path, assets.global_layout)
        self.cache = assets.load_cache(board)
        self.branch_tensors, self.self_kv_by_branch = {}, {}
        for condition in CONDITIONS:
            tensors = dict(self.tensors)
            if condition == 'positive':
                tensors.update({name: self.tensors['positive_'+name] for name in SCRATCH_NAMES})
            self.branch_tensors[condition] = tensors
            layout = PackedKVLayout(4096, 10)
            buffers = {name: board.make((size//4,), '<f4') if name in ('ks', 'vs')
                       else board.make((size,), 'u1') for name, size in layout.sizes.items()}
            self.self_kv_by_branch[condition] = SimpleNamespace(layout=layout, buffers=buffers,
                ready=False, vs_policy='PV_PROB_SCALE_FOLDED_V1')
        self.tiles = BranchSlotPools()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fixed-tp')
        self.sp_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fixed-sp')
        self.step, self.layer, self.condition, self.state = -1, -1, None, None
        self.stage_records = []
        self.execution_steps = assets.updates
        self.prefix_ready = False
        self.branch_states = {'positive': self.tensors['state1'], 'negative': self.tensors['state0']}
        self.schedule = SCHEDULE
        # Populate only this board's owned file pages after all live DMA/CMA
        # allocations exist. All boards prepare concurrently before RELEASE;
        # this measured preparation is separate from, and not hidden in, the
        # one-step runtime. The original two hot weight slots remain unchanged.
        self.weight_file_prefetch = OwnedWeightFilePrefetch(assets, cancel_check=cancel_check)
        self.weight_file_prefetch_report = self.weight_file_prefetch.prepare()
        self._plan_step=None;self._layer_plans={}
        self._build_step_plans(0)

    def _build_step_plans(self,step):
        """Bind immutable geometry/addresses once, before the first RELEASE.

        No tensor contents are read and no core/DMA is launched. Subsequent
        sampler steps rebind their different coefficient addresses explicitly.
        """
        if self.tiles.busy or self.weights.active is not None:raise RuntimeError('Cannot replace plans with live tensor readers')
        plans={}
        for layer in range(36):
            layout=self.assets.layers[layer][1]
            pack=weight_views(self.weights.slots[layer%2],layout)
            plan=LayerProgramPlan((self.generation,step,layer))
            for condition in CONDITIONS:
                t=self.branch_tensors[condition];state=self.branch_states[condition]
                ws=self.self_kv_by_branch[condition]
                states=[state]+([self.tensors['state0']] if layer==0 and condition=='positive' else [])
                for source in states:
                    for operator in range(3):
                        shift,gain,_=self.assets.modulation(step,layer,operator)
                        plan.add((condition,'mod',operator,source.address),[self.fixed_jobs.norm(source,t['quant'],shift,gain,t['sa'],scale_policy=self.assets.scale_policy(step,layer,operator))])
                plan.add((condition,'qkv'),[self._gemm_for(t,t[q],self._weight(pack,'self_attn.'+q+'_proj.weight'),self.role.rows,1280,5120) for q in ('q','k','v')])
                plan.add((condition,'rope_pack'),[self.fixed_jobs.rms_rope(t[q],pack['self_attn.'+q+'_norm.weight'],t['cos'],t['sin']) for q in ('q','k')]+[self.fixed_jobs.prepare_kv(t['k'],t['v'],ws)])
                plan.add((condition,'cross_q'),[self._gemm_for(t,t['q'],self._weight(pack,'cross_attn.q_proj.weight'),self.role.rows,1280,5120),bind_job(dict(stage=4,rows=self.role.rows*10,cols=128,lda=128,ldc=128),dict(a=t['q'],c=t['q'],aux0=pack['cross_attn.q_norm.weight']))])
                for cross in (False,True):
                    workspace=self.assets.prepared(condition,layer) if cross else ws
                    b,l=workspace.buffers,workspace.layout
                    plan.add((condition,'attention',cross),[bind_job(dict(stage=7,rows=self.role.rows,cols=512 if cross else 4096,heads=10,k_count=l.tokens,flags=160 if cross else 128,lda=1280,ldb=1280,ldc=1280),dict(a=t['q'],b=PackedKVBinding(b['k'],l,'k'),c=t['attn'],aux0=b['ks'],aux1=b['vs'],aux2=PackedKVBinding(b['v'],l,'v')))])
                # Layer0 negative cross reads the positive shared Q until both
                # complete. It needs its own immutable address binding.
                if layer==0 and condition=='negative':
                    w=self.assets.prepared(condition,layer);b,l=w.buffers,w.layout
                    plan.add((condition,'shared_cross'),[bind_job(dict(stage=7,rows=self.role.rows,cols=512,heads=10,k_count=l.tokens,flags=160,lda=1280,ldb=1280,ldc=1280),dict(a=self.branch_tensors['positive']['q'],b=PackedKVBinding(b['k'],l,'k'),c=t['attn'],aux0=b['ks'],aux1=b['vs'],aux2=PackedKVBinding(b['v'],l,'v')))])
                plan.add((condition,'quant'),[bind_job(dict(stage=2,rows=self.role.rows,cols=1280,lda=1280,ldc=1280),dict(a=t['attn'],c=t['quant'],aux0=t['sa']))])
                plan.add((condition,'ffn_up'),[self.fixed_jobs.ffn_up(t['quant'],self._weight(pack,'mlp.layer1.weight'),t['ffn'],t['sa'],t['ffn_scales'])])
                for operator,name,k in ((0,'self_attn.output_proj.weight',1280),(1,'cross_attn.output_proj.weight',1280),(2,'mlp.layer2.weight',5120)):
                    for first,rows in tiles(self.role.rows):
                        for local_slot in range(2):
                            slot=CONDITIONS.index(condition)*2+local_slot;out=self.tensors['partial'+str(slot)]
                            job=self._gemm_for(t,out,self._weight(pack,name),rows,5120,k,first=first,partial=True) if operator!=2 else self.fixed_jobs.ffn_down(t['ffn'],self._weight(pack,name),out,t['ffn_scales'],first,rows)
                            plan.add((condition,'projection',operator,first,slot),[job])
            plans[layer]=plan.seal()
            if hasattr(self,'b'):self.b.bind_fixed_plan(plans[layer])
        self._layer_plans=plans;self._plan_step=step

    def _fixed(self,ctx,key):
        if (ctx.generation,ctx.step,ctx.layer)!=(self.generation,self._plan_step,self.layer):raise RuntimeError('Fixed plan owner/context changed')
        return self._layer_plans[ctx.layer][(ctx.condition,)+key]

    def _activate(self, ctx):
        if threading.get_ident() != self._compute_thread:
            raise RuntimeError('Only the fixed compute scheduler may submit PL programs')
        # Reporting fields only: native reader jobs never consult these.
        self.step, self.layer, self.condition, self.state = ctx.step, ctx.layer, ctx.condition, ctx.state

    def _check_futures(self):
        for future in self._futures:
            if future.done():
                future.result()

    def asynchronous_ownership(self):
        return dict(pending_async_tasks=sum(not future.done() for future in self._futures),
                    tile_readers=int(self.tiles.busy))

    def _program(self, ctx, jobs):
        self._activate(ctx)
        self._check_futures()
        return self.b.program(jobs)

    def _make_contexts(self, step, layer):
        if self.tiles.busy:
            raise RuntimeError('Previous layer still owns TP slots')
        contexts = {}
        for condition in CONDITIONS:
            gates = []
            for operator in range(3):
                gate = self.tensors[f'{condition}_gate{operator}']
                source = self.assets.modulation(step, layer, operator)[2]
                gate.array[:] = 0
                gate.array.view('u1')[:source.nbytes] = source.array.view('u1')
                gate.root.flush()
                gates.append(gate)
            contexts[condition] = BranchContext.create(self.generation, step, layer, condition,
                self.branch_states[condition], self.branch_tensors[condition],
                self.self_kv_by_branch[condition], gates)
        return contexts

    def _gemm_for(self, tensors, out, weight, m, n, k, *, first=0, partial=False):
        return bind_job(dict(stage=1, rows=m, cols=n, inner=k, lda=k, ldb=k, ldc=n,
            flags=72 if partial else 64, a=tensors['quant'].address+first*k*2,
            aux0=tensors['sa'].address+first*4),
            dict(a=tensors['quant'], b=weight[0], c=out, aux0=tensors['sa'], aux1=weight[1]))

    def _quantize_for(self, ctx, source, columns):
        t = ctx.tensors
        return self._program(ctx, [bind_job(dict(stage=2, rows=self.role.rows,
            cols=columns, lda=source.shape[-1], ldc=columns),
            dict(a=source, c=t['quant'], aux0=t['sa']))])

    def _modulate(self,ctx,operator,*,state=None):
        source=ctx.state if state is None else state
        return self._program(ctx,self._fixed(ctx,('mod',operator,source.address)))

    def _prepare_self(self,ctx,pack):
        self._modulate(ctx,0)
        self._program(ctx,self._fixed(ctx,('qkv',)))
        self._program(ctx,self._fixed(ctx,('rope_pack',)))
        for value in ctx.workspace.buffers.values():value.root.invalidate()
        ctx.workspace.ready=False
        future=self.sp_pool.submit(self._exchange_kv,ctx);self._futures.append(future)
        return future

    def _exchange_kv(self, ctx):
        try:
            result = self.transport.allgather_kv(ctx.workspace, step=ctx.step,
                layer=ctx.layer, condition=ctx.condition)
            ctx.workspace.ready = True
            return result
        except BaseException as error:
            self._poison(error)
            raise

    def _attention_body(self, ctx, pack, *, cross=False, query=None, source_state=None,
                        output_state=None, before_projection=None):
        t, n = ctx.tensors, self.role.rows
        if cross:
            if query is None:
                self._modulate(ctx, 1, state=source_state)
                self._program(ctx,self._fixed(ctx,('cross_q',)))
                query = t['q']
            workspace = self.assets.prepared(ctx.condition, ctx.layer)
            logical = 512
        else:
            workspace, logical, query = ctx.workspace, 4096, t['q']
            if not workspace.ready:
                raise RuntimeError('Self attention cannot read a live KV DMA destination')
        if workspace.vs_policy != 'PV_PROB_SCALE_FOLDED_V1':
            raise RuntimeError('Attention VS representation differs from the overlap image')
        key=('shared_cross',) if cross and ctx.layer==0 and ctx.condition=='negative' else ('attention',cross)
        self._program(ctx,self._fixed(ctx,key))
        # The projection's two-slot producer can wait for a remote TP rank.
        # Publish next-layer weight prefetch before entering that wait, while
        # keeping this callback on the sole compute/weight-lifetime owner.
        if before_projection is not None:
            before_projection()
        return self._project_reduce(ctx, t['attn'], pack,
            ('cross_attn' if cross else 'self_attn')+'.output_proj.weight', 1280,
            1 if cross else 0, source_state=source_state, output_state=output_state)

    def _project_reduce(self,ctx,source,pack,name,k,operator,*,f16_scales=None,source_state=None,output_state=None):
        if f16_scales is None:self._program(ctx,self._fixed(ctx,('quant',)))
        state=ctx.state if source_state is None else source_state
        destination=ctx.state if output_state is None else output_state
        leases=[];future=None;native=None
        try:
            # Reserve both branch slots for this bounded native phase. The
            # queue reuses them only after actual two-round TP retirement.
            for first,_ in list(tiles(self.role.rows))[:2]:leases.append(self.tiles.acquire(ctx,operator,first))
            windows=list(tiles(self.role.rows));commands=[]
            for i,(first,rows) in enumerate(windows):commands.append(self._fixed(ctx,('projection',operator,first,leases[i&1][0])))
            partials=[self.tensors['partial'+str(slot)] for slot,_ in leases]
            sums=[self.tensors['sum'+str(slot)] for slot,_ in leases]
            native=self.b.prepare_projection(commands,partials,sums,state,destination,ctx.gates[operator],first_rows=[x[0] for x in windows],rows=[x[1] for x in windows],step=ctx.step,condition=ctx.condition,layer=ctx.layer,operator=operator,generation=ctx.generation)
            def consume():
                try:
                    result=self.b.reduce_projection(native)
                    for _,token in leases:self.tiles.release(ctx.condition,token)
                    return result
                except BaseException as error:self._poison(error);raise
            future=self.pool.submit(consume);self._futures.append(future)
            self.b.compute_projection(native)
            return ProjectionCompletion(ctx,operator,(future,))
        except BaseException as error:
            if native is not None:self.b.lib.cosmos_fixed_project_cancel(__import__('ctypes').byref(native),-15)
            self._poison(error)
            if future is not None:
                try:future.result()
                except BaseException as reduction_error:
                    # Keep the original reducer cause; compute -15 is only
                    # the cooperative cancellation propagated from that cause.
                    if getattr(native,'reduction_error',None) is not None:
                        raise RuntimeError('Projection reducer failed: '+repr(native.reduction_error)) from error
                    if 'compute failed -15' in str(error):
                        raise RuntimeError('Projection consumer failed: '+repr(reduction_error)) from error
            raise

    def _consume_tile(self, ctx, operator, first, rows, partial, summed, state,
                      destination, gate, token):
        try:
            source = View(state, first*5120*4, (rows, 5120), '<f4')
            output = View(destination, first*5120*4, (rows, 5120), '<f4')
            result = self.b.allreduce_tile(partial, summed, source, gate,
                rows=rows, first_row=first, step=ctx.step, condition=ctx.condition,
                layer=ctx.layer, operator=operator, generation=ctx.generation, output_state=output)
        except BaseException as error:
            self._poison(error)
            raise
        self.tiles.release(ctx.condition, token)
        return result

    def _mlp(self, ctx, pack, dependency):
        dependency.wait()
        self._modulate(ctx, 2)
        t = ctx.tensors
        self._program(ctx,self._fixed(ctx,('ffn_up',)))
        return self._project_reduce(ctx, t['ffn'], pack, 'mlp.layer2.weight', 5120, 2,
                                    f16_scales=t['ffn_scales'])

    def _prefetch_after_kv(self, step, layer):
        self.weights.self_kv_complete('positive', step=step, layer=layer)
        self.weights.self_kv_complete('negative', step=step, layer=layer)
        self.weights.prefetch_after_self(step=step, layer=layer, execution_steps=self.execution_steps)

    def common_prefix(self, pack, *, step, contexts=None):
        contexts = self._make_contexts(step, 0) if contexts is None else contexts
        ctx = replace(contexts['positive'], state=self.tensors['state0'])
        began = time.monotonic()
        self._prepare_self(ctx, pack).result()
        self._prefetch_after_kv(step, 0)
        self._attention_body(ctx, pack, output_state=self.tensors['state0']).wait()
        self._modulate(ctx, 1)
        t, n = ctx.tensors, self.role.rows
        self._program(ctx,self._fixed(ctx,('cross_q',)))
        self.prefix_ready, self.prefix_step = True, step
        self.shared_cross_query = t['q']
        return dict(step=step, layer=0, condition='common_prefix', seconds=time.monotonic()-began,
            rows=n, shared_patch=True, shared_self_attention=True, shared_cross_query=True,
            branch_fork='FIRST_POSITIVE_CROSS_RESIDUAL_OUT_OF_PLACE', cpu_hidden_fork_bytes=0)

    def _run_layer_pair(self, pack, *, step, layer, record=None):
        self.step,self.layer=step,layer
        self.b.install_fixed_plan(self._layer_plans[layer])
        contexts = self._make_contexts(step, layer)
        p, n = contexts['positive'], contexts['negative']
        began = time.monotonic()
        if layer == 0:
            common = self.common_prefix(pack, step=step, contexts=contexts)
            if record is not None:
                record(common)
            # Shared Q retains two readers until both cross programs finish.
            # P FUSED precedes N FUSED on one queue, so state0 is not overwritten
            # by N until every P reader of the shared prefix has completed.
            pcross = self._attention_body(p, pack, cross=True, query=self.shared_cross_query,
                source_state=self.tensors['state0'], output_state=p.state)
            ncross = self._attention_body(n, pack, cross=True, query=self.shared_cross_query,
                source_state=self.tensors['state0'], output_state=n.state)
        else:
            pkv = self._prepare_self(p, pack)
            nkv = self._prepare_self(n, pack)
            pkv.result()
            def prefetch_before_positive_projection():
                # P attention overlaps N KV. Only after N KV has drained may
                # P2/P3 carry the next weight transaction, in the old order.
                nkv.result()
                self._prefetch_after_kv(step, layer)
            pself = self._attention_body(p, pack,
                before_projection=prefetch_before_positive_projection)
            nself = self._attention_body(n, pack)
            pself.wait()
            pcross = self._attention_body(p, pack, cross=True)
            nself.wait()
            ncross = self._attention_body(n, pack, cross=True)
        pmlp = self._mlp(p, pack, pcross)
        nmlp = self._mlp(n, pack, ncross)
        pmlp.wait()
        self.weights.branch_complete('positive')
        nmlp.wait()
        self.weights.branch_complete('negative')
        if self.tiles.busy:
            raise RuntimeError('Layer complete while a TP reader still owns a slot')
        elapsed = time.monotonic()-began
        rows = [dict(step=step, layer=layer, condition=condition, seconds=elapsed,
            rows=self.role.rows, schedule=SCHEDULE, duration_scope='SHARED_PAIR_WINDOW_NOT_ADDITIVE')
            for condition in CONDITIONS]
        if record is not None:
            for row in rows:
                record(row)
        return rows

    def initialize(self, patch, *, step=None):
        if threading.get_ident()!=self._compute_thread:raise RuntimeError('Compute owner changed')
        if self.tiles.busy or self.weights.active is not None:
            raise RuntimeError('Previous step still owns tensor or weight readers')
        self.step = self.step+1 if step is None else step
        self.layer, self.condition = -1, 'common_prefix'
        if not 0 <= self.step < self.assets.updates:
            raise ValueError('Initial patch outside actual schedule')
        if patch.shape != (self.role.rows, 80) or patch.dtype != np.dtype('<f4'):
            raise ValueError('SP3 local patch shape')
        if np.any(patch[:,64:]):raise ValueError('Fixed sampler patch requires structural A zeros in columns64..79')
        if self._plan_step!=self.step:self._build_step_plans(self.step)
        t = self.tensors
        t['input'].array[:] = patch
        t['input'].root.flush()
        self.b.program([bind_job(dict(stage=2, rows=self.role.rows, cols=68, lda=80, ldc=80),
            dict(a=t['input'], c=t['quant'], aux0=t['sa']))])
        self.b.program([self._gemm_for(t, t['state0'], self._weight(self.globals, 'patch'),
            self.role.rows, 5120, 80)])
        self.prefix_ready = False
        self.prefix_step = None
        self.initialized_step = self.step

    def run_step(self, step, record=None):
        if not 0 <= step < self.assets.updates or getattr(self, 'initialized_step', None) != step \
                or getattr(self, 'completed_step', None) == step:
            raise RuntimeError('Each valid step requires its own fresh common patch')
        rows = []
        try:
            for layer in range(36):
                self._check_futures()
                pack = self.weights.acquire(layer, step=step)
                rows.extend(self._run_layer_pair(pack, step=step, layer=layer, record=record))
                self.weights.release()
                self._futures.clear()
            self.weights.join_prefetch()
        except BaseException as error:
            self._poison(error)
            raise
        self.completed_step = step
        self.stage_records.extend(rows)
        return rows

    def finish(self, condition, step):
        if threading.get_ident()!=self._compute_thread:raise RuntimeError('Compute owner changed')
        if condition not in CONDITIONS or self.tiles.busy or self.weights.active is not None:
            raise RuntimeError('Final projection requires drained branch state')
        self.step, self.layer, self.condition, self.state = step, 36, condition, self.branch_states[condition]
        t = self.branch_tensors[condition]
        shift, gain = self.assets.final(step)
        self.b.program([self.fixed_jobs.norm(self.state, t['quant'], shift, gain, t['sa'],
            scale_policy=self.assets.final_scale_policy(step))])
        self.b.program([self._gemm_for(t, self.tensors['prediction'], self._weight(self.globals, 'final'),
            self.role.rows, 64, 5120)])
        return self.tensors['prediction']

    def _poison(self, error):
        if hasattr(self, 'tiles'):
            self.tiles.fail(error)
        self.b.poisoned = True
        self.b.cancel_native()

    def close(self):
        self.join_workers()
        if self.tiles.busy:
            raise RuntimeError('Model still owns a live or failed DMA slot')
        self.weights.close()

    def join_workers(self):
        """Join all attempted native tasks before any external DMA/CMA cleanup."""
        for pool in (self.sp_pool, self.pool):
            if pool is not None:
                pool.shutdown(wait=True)
        if self.weights is not None:
            self.weights.network.shutdown(wait=True)
            self.weights.executor.shutdown(wait=True)
