"""Hardware leases, native DMA structures and engine ownership."""
from pathlib import Path
import ctypes as C
import numpy as np
import threading,os,time,struct
from board_driver import Board,CoreJob,Stats,RegWrite

HARDWARE_LOCK = '/var/lock/cosmos2b_fpga.lock'
_hardware_lease_guard = threading.RLock()
_hardware_lease_file = None
_hardware_lease_pid = os.getpid()
_hardware_leases = {}


class HardwareLease:
    """Process-local nesting around one nonblocking, board-wide Linux flock."""
    def __init__(self, purpose):
        self.purpose, self.closed = purpose, False

    def close(self):
        global _hardware_lease_file
        with _hardware_lease_guard:
            if self.closed: return
            _hardware_leases.pop(id(self), None)
            self.closed = True
            if not _hardware_leases and _hardware_lease_file is not None:
                _hardware_lease_file.close()
                _hardware_lease_file = None


def acquire_hardware_lease(purpose='board'):
    """Probe may wrap one Board instance; independent Boards cannot coexist."""
    global _hardware_lease_file, _hardware_lease_pid
    if purpose not in ('board', 'probe'): raise ValueError('Unknown C206 hardware lease purpose')
    with _hardware_lease_guard:
        if _hardware_lease_pid != os.getpid():
            # A fork must not inherit permission to touch its parent's board.
            # Closing only this duplicate fd does not unlock the parent's fd.
            if _hardware_lease_file is not None: _hardware_lease_file.close()
            _hardware_lease_file, _hardware_lease_pid = None, os.getpid()
            _hardware_leases.clear()
        holders = list(_hardware_leases.values())
        if (purpose == 'probe' and holders) or any(kind == 'board' for kind, _ in holders):
            raise RuntimeError('Another C206 Board/probe already owns this process hardware lease')
        if any(owner != threading.get_ident() for _, owner in holders):
            raise RuntimeError('C206 probe and Board must have the same initialization owner')
        if _hardware_lease_file is None:
            import fcntl
            handle = open(HARDWARE_LOCK, 'a+')
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise RuntimeError('Another process owns '+HARDWARE_LOCK)
            _hardware_lease_file = handle
        lease = HardwareLease(purpose)
        _hardware_leases[id(lease)] = (purpose, threading.get_ident())
        return lease


class StreamLane(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ('dma', 'residual_dma', 'link', 'reducer')]


class StreamReduceJob(C.Structure):
    _fields_ = [('lanes', StreamLane*3)] + [(name, C.c_uint64) for name in
                ('local_phys', 'state_phys', 'gate_phys')] + [(name, C.c_uint32) for name in
                ('bytes', 'cols', 'epoch', 'step', 'branch', 'layer', 'operator_id',
                 'group', 'node', 'peer', 'timeout_ms')]
    _fields_ += [('first_row', C.c_uint32), ('preload_gate', C.c_uint32)]


class StreamReduceStats(C.Structure):
    _fields_ = [('elapsed_ns', C.c_uint64)] + [(name, C.c_uint32) for name in
                ('error', 'chunks', 'rx_bytes', 'tx_bytes')]


class BulkJob(C.Structure):
    _fields_ = [('lane', StreamLane), ('tx_phys', C.c_uint64), ('rx_phys', C.c_uint64)]
    _fields_ += [(name, C.c_uint32) for name in
                ('bytes', 'node', 'peer', 'lane_id', 'job_id', 'step', 'channel', 'sequence', 'timeout_ms')]


NEIGHBOURS = {1: (2, 3, 4), 2: (1, 4, 3), 3: (4, 1, 2), 4: (3, 2, 1)}

class C206Board(Board):
    def _init_concurrency(self):
        self._core_lock, self._collective_lock, self._state_lock = threading.Lock(), threading.Lock(), threading.RLock()
        self._bulk_guard = threading.RLock()
        self._bulk_users, self._bulk_failed = 0, False
        self._lane_locks = [threading.Lock() for _ in range(3)]
        self._bulk_buffers = {}
        self._used_bulk_contexts = set()
        self.transport = self
        self._inflight = 0
        self._overlap_events = 0
        # Conservative lifetime record: mark before entering native code so a
        # timeout/exception cannot hide an attempted DMA from safe shutdown.
        self._dma_attempted = set()

    def _check_available(self):
        if getattr(self, 'poisoned', False) or getattr(self, 'needs_reinitialization', False):
            raise RuntimeError('C206 failed fabric requires full PL recovery and a new driver before retry')

    def _begin_operation(self):
        # __new__ based host fakes may initialize lazily; real construction does
        # this before either producer/consumer can run.
        if not hasattr(self, '_state_lock'): self._init_concurrency()
        with self._state_lock:
            self._check_available()
            marker = self._overlap_events
            if self._inflight: self._overlap_events += 1
            self._inflight += 1
            self.running = True
            return marker

    def _end_operation(self, failed):
        with self._state_lock:
            self._inflight -= 1
            if failed: self.poisoned = True
            self.running = bool(self._inflight or getattr(self, 'poisoned', False))

    def invoke(self, **args):
        if not hasattr(self, '_core_lock'): self._init_concurrency()
        with self._core_lock:
            overlap_marker = self._begin_operation()
            failed = True
            try:
                defaults = dict(act_offset=0, weights_offset=0, output_r_offset=0, aux_offset=0,
                    stage=0, rows=1, cols=8, inner=0, lda=8, ldb=8, ldc=8, heads=0, flags=0,
                    mode=0, k_start=0, k_count=0, a=0, b=0, c=0, aux0=0, aux1=0,
                    aux2=0, aux3=0, factor=0., epsilon=1e-6)
                defaults.update(args)
                if self.protocol.get('compute_control_contract')=='FIVE_MINIMAL_CSR_V1':
                    # The fixed program validator already owns every job's
                    # geometry and address bounds. These legacy envelope
                    # values are host assertions, not surviving PL registers.
                    if (int(defaults['stage'])!=11 or not 1<=int(defaults['rows'])<=128
                        or int(defaults['heads']) not in (1,2,3,4)
                        or int(defaults['mode'])!=int(self.protocol['hardware_id'],16)
                        or int(defaults['flags']) or int(defaults['aux0'])&31):
                        raise ValueError('Fixed minimal CSR program envelope')
                    defaults={name:defaults[name] for name in
                        ('act_offset','weights_offset','output_r_offset','aux_offset','rows','aux0')}
                writes, pointers = [], set(self.protocol['pointer_registers'])
                for name, value in defaults.items():
                    offset = self.regs[name]
                    if name in ('factor', 'epsilon'): value = struct.unpack('<I', struct.pack('<f', float(value)))[0]
                    value = int(value)
                    writes.append((offset, value & 0xffffffff))
                    if name in pointers: writes.append((offset+4, value >> 32))
                settings = (RegWrite*len(writes))(*(RegWrite(*entry) for entry in writes))
                job = CoreJob(self.core.array.ctypes.data, settings, len(writes),
                              self.regs['error'], self.regs['completed_tiles'], 120000)
                stat = Stats()
                code = self.lib.cosmos_stage(C.byref(job), C.byref(stat))
                if code or stat.error:
                    raise RuntimeError('Core code=%d error=%d ctrl=%x' % (code, stat.error, self.core.read(0)))
                failed = False
                return dict(ms=stat.elapsed_ns/1e6, tiles=stat.tiles, error=stat.error,
                            host_operations_overlapped=self._overlap_events > overlap_marker)
            finally:
                self._end_operation(failed)

    def snapshot(self):
        interfaces = [dict(lane=lane['lane'], identity=hex(lane['link'].read(0x7c)),
                          node=lane['link'].read(0x90), peer=lane['link'].read(0x94), group=lane['link'].read(0x9c),
                          link=lane['link'].read(4), phy=lane['link'].read(0x0c),
                          rx_errors=lane['link'].read(0x4c), mac_faults=lane['link'].read(0x50),
                          router_drops=lane['link'].read(0xd4), router_relays=lane['link'].read(0xd8),
                          router_delivered=lane['link'].read(0xdc),
                          reducer_status=lane['reducer'].read(4), reducer_error=lane['reducer'].read(0x18),
                          dma=[lane['dma'].read(4), lane['dma'].read(0x34)]) for lane in self.interfaces]
        return dict(role=self.role, node=self.node, pair_rank=self.rank, group=self.group, peer=self.peer,
                    clock_mhz=self.clock, core=self.core.read(0), identity=hex(self.link.read(0x7c)),
                    interface_count=3, interfaces=interfaces,
                    rx_errors=sum(lane['rx_errors'] for lane in interfaces),
                    router_drops=max(lane['router_drops'] for lane in interfaces),
                    mac_faults=sum(lane['mac_faults'] for lane in interfaces))

