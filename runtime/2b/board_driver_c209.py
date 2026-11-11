"""Serialize on-board descriptor publication."""
import threading
from board_driver_c206 import C206Board

class C209Board(C206Board):
    def _init_concurrency(self):
        super()._init_concurrency()
        # Bulk phases join across threads: the last participant releases the
        # compute fence acquired by the first, so that fence must stay Lock.
        # Descriptor publication has its own same-thread nesting in the fixed
        # wrapper; inherited invoke acquires the compute fence exactly once.
        self._program_lock=threading.RLock()

