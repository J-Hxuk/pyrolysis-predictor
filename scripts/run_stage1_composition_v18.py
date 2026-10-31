"""CLI: Stage 1 BET composition incremental-value experiment (data/model read-only)."""
import argparse
from pathlib import Path
from pyrolysis.biochar_stage1.comp_ablation_v18 import run


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--outdir',type=Path,default=None)
    p.add_argument('--bootstrap',type=int,default=2500)
    args=p.parse_args()
    dest=args.outdir or args.root / 'reports/stage1_composition_v18'
    result=run(args.root,dest,repeats=args.bootstrap)
    print('Output:',dest)
    for m in result['metrics']:
        print(f"{m['algorithm']:12} {m['feature_set']:26} LODO R2={m['r2_oof']:.3f} MAE={m['mae_m2_g']:.2f} m2/g")


if __name__ == '__main__':
    main()
