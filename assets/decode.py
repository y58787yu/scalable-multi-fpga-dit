"""Decode a completed Cosmos image latent to a PNG."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent",type=Path,required=True,help="Final FP32 .npy")
    parser.add_argument("--vae",type=Path,required=True,help="Diffusers Cosmos AutoencoderKLWan directory")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--cpu-threads",type=int,default=2,help="CPU threads used for VAE decoding")
    args=parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    a=np.load(args.latent,allow_pickle=False)
    if a.shape!=(1,16,1,128,128) or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError("Expected a finite FP32 [1,16,1,128,128] image latent")
    vae=AutoencoderKLWan.from_pretrained(args.vae,torch_dtype=torch.float32,local_files_only=True).eval().to(args.device)
    vae.disable_tiling()
    vae.disable_slicing()
    mean=torch.tensor(vae.config.latents_mean,dtype=torch.float32).reshape(1,16,1,1,1)
    std=torch.tensor(vae.config.latents_std,dtype=torch.float32).reshape_as(mean)
    with torch.inference_mode():
        pixels=vae.decode((torch.from_numpy(a)*std+mean).to(args.device),return_dict=False)[0]
        if pixels.shape!=(1,3,1,1024,1024) or not torch.isfinite(pixels).all():
            raise ValueError("Unexpected VAE output")
        rgb=(pixels.float().cpu()[0,:,0].permute(1,2,0).clamp(-1,1)*.5+.5).mul(255).round().byte().numpy()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb).save(args.out,format="PNG")
if __name__=="__main__":
    main()
