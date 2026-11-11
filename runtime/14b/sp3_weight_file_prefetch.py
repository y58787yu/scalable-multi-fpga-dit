"""Prepare only this board's owned raw weight pages before the timed update.

The existing two DMA/CMA hot slots and all tensor ownership are unchanged.
This uses one bounded userspace read buffer and ordinary reclaimable Linux
file cache.  It neither copies weights into a new resident tensor nor pins
pages.  Preparation time is reported separately and belongs to cold-start
image latency, even when the subsequent compute interval is measured alone.
"""
import ctypes
import mmap
import os
from pathlib import Path
import time


def _memory():
    values = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'Cached', 'CmaTotal', 'CmaFree'):
                values[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def _storage_reads():
    try:
        values = dict(line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())
        return int(values['read_bytes'])
    except (OSError, ValueError, KeyError):
        return None


def _merge(ranges):
    result = []
    for begin, end in sorted(ranges):
        if begin < 0 or end <= begin:
            raise ValueError('Invalid owned weight byte range')
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((begin, end))
    return result


def _resident(path, ranges, page_size):
    """One mincore snapshot; no data faults, touching or page locking."""
    pages = _merge([(a // page_size, (b + page_size - 1) // page_size) for a, b in ranges])
    required = sum(b - a for a, b in pages)
    try:
        library = ctypes.CDLL(None, use_errno=True)
        mincore = library.mincore
        mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte))
        mincore.restype = ctypes.c_int
        with path.open('rb') as stream:
            size = os.fstat(stream.fileno()).st_size
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
                anchor = ctypes.c_char.from_buffer(mapping)
                address = ctypes.addressof(anchor)
                del anchor
                count = (size + page_size - 1) // page_size
                vector = (ctypes.c_ubyte * count)()
                if mincore(address, size, vector):
                    raise OSError(ctypes.get_errno(), 'mincore')
                resident = sum(int(bool(vector[i] & 1)) for a, b in pages for i in range(a, b))
        return dict(available=True, required_pages=required, resident_pages=resident,
                    missing_pages=required-resident)
    except (AttributeError, OSError, ValueError, BufferError) as error:
        return dict(available=False, required_pages=required, error=str(error))


class OwnedWeightFilePrefetch:
    def __init__(self, assets, *, chunk_bytes=4*1024**2, reserve_bytes=256*1024**2,
                 cancel_check=None):
        if not isinstance(chunk_bytes, int) or not 4096 <= chunk_bytes <= 16*1024**2:
            raise ValueError('Owned weight prefetch buffer bounds')
        if not isinstance(reserve_bytes, int) or reserve_bytes < 0:
            raise ValueError('Owned weight prefetch reserve')
        self.assets, self.chunk_bytes = assets, chunk_bytes
        self.reserve_bytes, self.cancel_check = reserve_bytes, cancel_check
        self.report = None

    def _check_cancel(self):
        if self.cancel_check is not None:
            self.cancel_check()

    def prepare(self):
        if self.report is not None:
            raise RuntimeError('Owned weight preparation is one-use')
        started = time.monotonic_ns()
        page_size = mmap.PAGESIZE
        owned = list(self.assets.role.owned_layers)
        expected = list(range(self.assets.role.group*12, (self.assets.role.group+1)*12))
        if owned != expected:
            raise ValueError('Owned weight prefetch requires the unchanged twelve-layer ownership')
        plan = []
        for layer in owned:
            self._check_cancel()
            source, layout = self.assets.layers[layer]
            path = Path(source).with_suffix('.bin')
            size = path.stat().st_size
            ranges = _merge([(entry['source_offset'], entry['source_offset']+entry['bytes'])
                             for entry in layout['arrays'].values()])
            if any(end > size for _, end in ranges):
                raise IOError('Owned raw weight file is smaller than the bound layout')
            before = _resident(path, ranges, page_size)
            plan.append(dict(layer=layer, path=path, ranges=ranges, file_bytes=size,
                             required_bytes=sum(b-a for a, b in ranges), residency_before=before))
        memory_before = _memory()
        required_pages = sum(entry['residency_before']['required_pages'] for entry in plan)
        required_bytes = sum(entry['required_bytes'] for entry in plan)
        available = memory_before.get('MemAvailable')
        report = dict(schema='OWNED_RAW_FILE_PREFETCH_V1', group=self.assets.role.group,
            rank=self.assets.role.rank, owned_layers=owned, remote_layers_read=0,
            userspace_buffer_bytes=self.chunk_bytes, added_cma_bytes=0,
            page_size=page_size, required_bytes=required_bytes,
            required_file_page_bytes=required_pages*page_size,
            memory_before_bytes=memory_before, reserve_bytes=self.reserve_bytes,
            budget_feasible=available is None or available >= required_pages*page_size+self.reserve_bytes,
            cache_is_reclaimable=True, pages_locked=False, preparation_included_in_cold_start=True,
            per_file=[], status='PREPARING', start_ns=started)
        self.report = report
        if not report['budget_feasible']:
            report.update(status='INSUFFICIENT_FILE_CACHE_BUDGET', elapsed_seconds=(time.monotonic_ns()-started)/1e9)
            raise MemoryError('Owned weight pages plus reserve exceed post-allocation MemAvailable')
        storage_before = _storage_reads()
        buffer = bytearray(self.chunk_bytes)
        view = memoryview(buffer)
        try:
            for entry in plan:
                self._check_cancel()
                began = time.monotonic_ns()
                read_bytes = 0
                # Read existing raw byte ranges only.  No hashing, conversion,
                # source modification, extra CMA allocation or DMA is involved.
                with entry['path'].open('rb', buffering=0) as stream:
                    for begin, end in entry['ranges']:
                        stream.seek(begin)
                        remaining = end-begin
                        while remaining:
                            self._check_cancel()
                            size = min(remaining, self.chunk_bytes)
                            count = stream.readinto(view[:size])
                            if count is None or count <= 0:
                                raise IOError('Truncated owned raw weight during preparation')
                            remaining -= count
                            read_bytes += count
                report['per_file'].append(dict(layer=entry['layer'], path=str(entry['path']),
                    file_bytes=entry['file_bytes'], required_bytes=entry['required_bytes'],
                    read_bytes=read_bytes, elapsed_seconds=(time.monotonic_ns()-began)/1e9,
                    residency_before=entry['residency_before']))
            # Snapshot all files only after the last read, so a later file
            # evicting earlier pages is visible.  This is not a residency lock.
            for entry, result in zip(plan, report['per_file']):
                self._check_cancel()
                result['residency_after'] = _resident(entry['path'], entry['ranges'], page_size)
            available_results = [x['residency_after'] for x in report['per_file']]
            measured = all(x['available'] for x in available_results)
            report['residency_measured'] = measured
            if measured:
                report['resident_pages_after'] = sum(x['resident_pages'] for x in available_results)
                report['missing_pages_after'] = sum(x['missing_pages'] for x in available_results)
                report['resident_fraction_after'] = report['resident_pages_after']/required_pages
                report['all_required_pages_resident_at_snapshot'] = report['missing_pages_after'] == 0
            report['status'] = 'READ_COMPLETE'
        except BaseException as error:
            report.update(status='FAILED', error=repr(error))
            raise
        finally:
            view.release()
            storage_after = _storage_reads()
            report['elapsed_seconds'] = (time.monotonic_ns()-started)/1e9
            report['end_ns'] = time.monotonic_ns()
            report['read_bytes'] = sum(x['read_bytes'] for x in report['per_file'])
            report['process_storage_read_bytes'] = None if storage_before is None or storage_after is None else storage_after-storage_before
            report['storage_counter_scope'] = 'PROCESS_WIDE_READ_BYTES_DELTA_NOT_EXCLUSIVE_THREAD_IO'
            report['memory_after_bytes'] = _memory()
        return report
