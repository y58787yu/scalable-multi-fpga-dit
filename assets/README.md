# Fixed image assets

`prepare.py` reads the official Cosmos-Predict2 Text2Image `model.pt`. It computes
INT16 weights and a scale per full output channel before assigning channels to
ranks. Each output directory `nodeN/` contains that rank's owned weight panels,
all layer metadata, fixed text K/V, modulation coefficients, RoPE and sampler inputs.
The source grid uses 10 updates for 2B and 5 or 10 updates for 14B. Provide a fresh
output directory on a filesystem with large-file support.

Use the 2B repository's `scheduler/` and `vae/` for both models. The fixed
schedule, image geometry and CPU reference seed are shared across the image groups.
The FPGA SiLU coefficient table is stored in `silu_u32.hex`. Weight, cache and input
identities for these fixed profiles are recorded in `identities.json`.

`apple_context.npz` contains the nonzero input embeddings for the fixed
positive and negative prompts, 16 and 11 rows of 1,024 values respectively,
encoded with the official T5 encoder. The archive stores the BF16-origin values
losslessly as FP16. To reproduce it, supply the shared repository's
`tokenizer/` and `text_encoder/` to `encode_prompt.py --components DIR --out FILE`
using BF16-capable CUDA. The two prompt strings are defined in `inputs.py`.

Official sources at the revisions matching these weights:
[2B and shared components](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image/tree/acdb5fde992a73ef0355f287977d002cbfd127e0)
and [14B](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Text2Image/tree/015332720f70dd7b497c1cff9fd0c936a77f160b).
