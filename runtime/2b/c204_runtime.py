"""Aligned weight-shard layout and typed DMA views."""
from pathlib import Path
import json
import zipfile
import numpy as np

ALIGNMENT = 64

def aligned(value, alignment=ALIGNMENT):
    return (int(value) + alignment - 1) // alignment * alignment

def weight_layout(path):
    """Read headers only: planning a 2 GiB model must not materialize it."""
    path = Path(path)
    raw = path.with_suffix('.bin')
    layout_file = path.with_suffix('.layout.json')
    if raw.exists():
        layout = json.loads(layout_file.read_text())
        if raw.stat().st_size != layout['bytes']:
            raise ValueError('Raw weight size differs from layout: ' + str(path))
    else:
        arrays, offset = {}, 0
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith('.npy'):
                    continue
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError('Unsupported NPY header version: '+str(version))
                if fortran or dtype.hasobject:
                    raise ValueError('Weights must be C-contiguous numeric arrays')
                offset = aligned(offset)
                arrays[name[:-4]] = dict(offset=offset, shape=list(shape), dtype=dtype.str)
                offset += int(np.prod(shape)) * dtype.itemsize
        layout = dict(bytes=aligned(offset), arrays=arrays)
    limit = int(layout['bytes'])
    ranges = []
    if limit <= 0 or limit % ALIGNMENT:
        raise ValueError('Weight allocation must be positive and 64-byte aligned')
    for name, entry in layout['arrays'].items():
        dtype = np.dtype(entry['dtype'])
        if dtype.hasobject or any(int(x) < 0 for x in entry['shape']):
            raise ValueError('Invalid weight array ' + name)
        first = int(entry['offset'])
        end = first + int(np.prod(entry['shape'])) * dtype.itemsize
        if first < 0 or first % ALIGNMENT or end > limit:
            raise ValueError('Invalid weight offset ' + name)
        ranges.append((first, end))
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError('Overlapping weight arrays')
    return layout

class WeightView:
    def __init__(self, parent, entry):
        self.root = parent.root
        offset = int(entry['offset'])
        self.address = parent.address + offset
        self.shape = tuple(entry['shape'])
        self.dtype = np.dtype(entry['dtype'])
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.array = parent.array.view('u1').reshape(-1)[offset:offset+self.nbytes].view(self.dtype).reshape(self.shape)
        if hasattr(parent, 'allocator'):
            parent.allocator.register_view(self.array)
