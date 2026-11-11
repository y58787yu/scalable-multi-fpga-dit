# Cosmos on ZCU102

Distributed inference for Cosmos-Predict2 text-to-image denoisers at 1024 × 1024. Boards share model weights and image tokens over direct SFP+ links. Each node uses five 256-lane matrix slices.

![Cosmos-Predict2-14B, 1024 × 1024, ten denoising updates](examples/apple.png)

*Cosmos-Predict2-14B · 10 denoising updates · seed 20261006*

| Model | Boards | Image groups × boards per group | Inventory |
| --- | ---: | --- | --- |
| Cosmos-Predict2-2B | 2 | 1 × 2 | [2b-2.json](examples/2b-2.json) |
| Cosmos-Predict2-2B | 4 | 2 × 2 | [2b-4.json](examples/2b-4.json) |
| Cosmos-Predict2-14B | 12 | 3 × 4 | [14b-12.json](examples/14b-12.json) |

The two 2B configurations use the same overlay. Each model's [runtime](runtime/) includes its bitstream, HWH, XRT metadata and native library. The [hardware build](hardware/BUILD.md) uses Vivado and Vitis HLS 2023.2. Third-party hardware notices are [here](hardware/THIRD_PARTY.md).

### Models and boards

Download the official [2B](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image/tree/acdb5fde992a73ef0355f287977d002cbfd127e0) or [14B](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Text2Image/tree/015332720f70dd7b497c1cff9fd0c936a77f160b) denoiser checkpoint (`model.pt`). Both configurations use the pinned 2B Diffusers components: use the parent of `scheduler/` for preparation and `vae/` for decoding. The input package includes the conditioning vectors for the fixed apple prompts. The seed selects the initial latent.

Use ZCU102 boards with 4 GB PS DDR, a 64-bit PYNQ Linux environment with XRT and NumPy, and a 1008 MiB CMA reservation. Set up SSH key authentication and the host keys on the PC. Copy the inventory you need to `cluster.local.json`. Set its hostnames, SSH user, remote install directory, PYNQ Python interpreter and four RX polarity bits per board. Set `sudo` to `true` if the SSH user uses passwordless sudo for overlay access.

SFP ports are numbered from zero. Groups and ranks are the logical coordinates in the inventory.

| Configuration | Fiber connections |
| --- | --- |
| 2B | Within each group, join rank 0 SFP1 to rank 1 SFP1. For four boards, also connect the two groups' matching ranks, SFP2 to SFP2. |
| 14B | Within each group, SFP0 connects ranks 0–1 and 2–3; SFP1 connects ranks 0–2 and 1–3. For each rank, connect group `g` SFP2 to group `(g+1) mod 3` SFP3. |

### Run

From the repository root on the PC with Python 3.11 or 3.12, using the 14B inventory:

```sh
python -m pip install -r requirements.txt
python assets/prepare.py --model 14b --groups 3 --steps 10 --weights /models/14b/model.pt --components /models/2b --out work/apple14b
python cluster.py --config cluster.local.json install --name apple14b10 --assets work/apple14b
python cluster.py --config cluster.local.json run --name apple14b10 --steps 10 --seed 20261006 --generation 1 --output work/latent.npy
python assets/decode.py --latent work/latent.npy --vae /models/2b/vae --out work/apple.png
```

For 2B, use its checkpoint and inventory, `--model 2b`, and `--groups 1` or `--groups 2` during preparation. Use distinct deployment names for each configuration and schedule. Preparation writes one asset directory per logical role. Installation copies each board's own assets and the model runtime. Increment the generation for each subsequent run. To export the first update only, use `run --steps 1`.
