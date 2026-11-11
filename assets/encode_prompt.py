"""Recreate the frozen apple text conditioning using the official T5 encoder."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components",type=Path,required=True,help="Model tree with tokenizer/ and text_encoder/")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from accelerate import cpu_offload
    from transformers import T5EncoderModel,T5TokenizerFast
    from inputs import PROMPTS,load_context
    tokenizer=T5TokenizerFast.from_pretrained(args.components/"tokenizer",local_files_only=True)
    encoder=T5EncoderModel.from_pretrained(args.components/"text_encoder",torch_dtype=torch.bfloat16,
                                          low_cpu_mem_usage=True,local_files_only=True).eval()
    cpu_offload(encoder,execution_device=torch.device("cuda:0"),offload_buffers=True)
    frozen=load_context()
    contexts={}
    with torch.inference_mode():
        for name,prompt in PROMPTS.items():
            tok=tokenizer(prompt,padding=False,max_length=512,truncation=True,return_tensors="pt")
            embedding=encoder(input_ids=tok.input_ids.to("cuda"),attention_mask=tok.attention_mask.to("cuda")).last_hidden_state
            value=embedding[0].float().cpu().numpy()
            if not np.array_equal(value, frozen[name]):
                raise ValueError(f"{name} differs from the frozen conditioning vectors")
            contexts[name]=value.astype("<f2")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,**contexts)
if __name__=="__main__":
    main()
