"""One owned process cancellation watcher; never resets DMA or frees memory."""
import ctypes
import json
from pathlib import Path
import threading

class SP3Cancelled(RuntimeError):pass

class SP3Cancellation:
    def __init__(self,root,job,nonce,*,native_cancel=None,poll_seconds=.5):
        if not isinstance(nonce,str) or not nonce:raise ValueError('Explicit owned job nonce required')
        self.path=Path(job)/'cancel.json';self.nonce=nonce;self.poll_seconds=poll_seconds
        self.requested=threading.Event();self.finished=threading.Event();self.thread=None
        if native_cancel is None:
            # Absolute path matches TP4Board. dlopen shares the existing .so
            # atomic cancellation flag, including network-prefetch native waits.
            self.lib=ctypes.CDLL(str(Path(root).resolve()/'libcosmos_tp4.so'))
            native_cancel=self.lib.cosmos_tp4_cancel
            native_cancel.argtypes=[ctypes.c_uint32];native_cancel.restype=None
        self.native_cancel=native_cancel
    def poll(self):
        if not self.requested.is_set():
            try:value=json.loads(self.path.read_text(encoding='utf-8'))
            except (OSError,ValueError):value=None
            if isinstance(value,dict) and value.get('nonce')==self.nonce:self.requested.set()
        if self.requested.is_set():self.native_cancel(1)
    def check(self):
        if self.requested.is_set():raise SP3Cancelled('Owned SP3 cancellation: '+self.nonce)
    def start(self):
        self.poll();self.check()
        def watch():
            while not self.finished.wait(self.poll_seconds):self.poll()
        self.thread=threading.Thread(target=watch,name='sp3-owned-cancel',daemon=True);self.thread.start()
    def close(self):
        self.finished.set()
        if self.thread is not None:self.thread.join()
