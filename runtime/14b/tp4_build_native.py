"""Build the ARM native executor on the deployment host; never programs PL."""
from pathlib import Path
import argparse
import subprocess
from tp4_profile import PROFILE

def header():
    values={'BIT_ID':int(PROFILE['hardware_id'],16),'RUNTIME_ID':int(PROFILE['runtime_id'],16),
        **{k.upper():PROFILE[k] for k in ('tokens','hidden','layers','updates','tensor_parallel','pipeline_stages','sequence_parallel','weight_groups','tile_rows','packet_max_bytes','boards')},
        'MAX_LOCAL_ROWS':max(PROFILE['sequence_rows']),'TAIL_ROWS':128}
    return '#ifndef TP4_CONTRACT_H\n#define TP4_CONTRACT_H\n/* Generated from config/tp4_profile.json. */\n'+''.join(f'#define TP4_{k} {v}U\n' for k,v in values.items())+'#endif\n'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--emit-header',action='store_true');parser.add_argument('--cc',default='gcc');parser.add_argument('--output');args=parser.parse_args()
    root=Path(__file__).resolve().parent;(root/'tp4_contract.h').write_text(header(),encoding='ascii')
    if not args.emit_header:
        output=Path(args.output) if args.output else root/'libcosmos_tp4.so'
        output.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([args.cc,'-O2','-std=c11','-fPIC','-shared','-D_POSIX_C_SOURCE=200809L','-ffp-contract=off','-o',str(output),str(root/'executor_tp4.c'),'-lm','-pthread'],check=True)
if __name__=='__main__':main()
