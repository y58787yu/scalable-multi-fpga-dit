# ZCU102 overlay build

Install Vitis HLS 2023.2 (SW build 4023990), Vivado 2023.2 (SW build 4029153), the ZCU102 board definition `xilinx.com:zcu102:part0:3.4`, Vitis `xclbinutil`, and Python 3.8 or newer. Set up the AMD environment so all three tools are on PATH. Run one of these commands from the repository root, choosing a fresh output directory.

```sh
python hardware/build.py --model 2b  --output build/cosmos-2b
python hardware/build.py --model 14b --output build/cosmos-14b
```

The device is `xczu9eg-ffvb1156-2-e`. The MMCM supplies separate system and MAC clocks. Each command exports its compute IP, applies the dual-clock and memory bindings, builds the Zynq MPSoC block design and writes `artifacts/cosmos_net.{bit,hwh,xclbin}`. Use `--ip-only` to build only the compute IP. Pair each overlay with its model's native runtime.

| Model | Compute IP | Physical SFP ports | Shell / runtime identity |
| --- | --- | --- | --- |
| [2B](2b/BUILD.md) | 20.0 | 1, 2 | 0xC2092B10 / 0xC2092B11 |
| [14B](14b/BUILD.md) | 16.0 | 0, 1, 2, 3 | 0xC2091000 / 0xC2091001 |

The source build runs fresh implementation. The supplied overlays are under `runtime/<model>/hardware/` at repository level.

Ethernet sources and licensing information are in [THIRD_PARTY.md](THIRD_PARTY.md).
