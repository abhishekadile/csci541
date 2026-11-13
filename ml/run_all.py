"""Full pipeline: split -> train A -> train B -> evaluate (val, then frozen test) -> compare.

    python -m ml.run_all                 # everything (long: up to 50 epochs per model)
    python -m ml.run_all --resume        # continue interrupted training
    python -m ml.run_all --batch-size 4  # GPU out of memory? keep resolution, lower batch size
    python -m ml.run_all --skip-train    # only (re)evaluate + compare

Models are trained one after the other with identical settings except the auxiliary loss weight.
"""
from __future__ import annotations

import argparse

from . import compare, split, train
from .config import load_config, run_dir, weights_dir
from .evaluate import evaluate, print_result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--num-workers", type=int, help="DataLoader workers for training (0 = Windows-safe default)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--models", nargs="+", default=["model_a", "model_b"], choices=["model_a", "model_b"])
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()

    import sys
    sys.argv = [sys.argv[0]]
    split.main()  # no-op if the manifest already exists
    for m in a.models:
        cfg_name = f"{m}.yaml"
        if not a.skip_train:
            targs = [cfg_name, "--device", a.device]
            if a.batch_size:
                targs += ["--batch-size", str(a.batch_size)]
            if a.max_epochs:
                targs += ["--max-epochs", str(a.max_epochs)]
            if a.num_workers is not None:
                targs += ["--num-workers", str(a.num_workers)]
            done = (run_dir(load_config(cfg_name)) / "run_summary.json").is_file()
            if a.resume and done:
                print(f"[{m}] already finished; skipping training")
            else:
                if a.resume and (weights_dir(load_config(cfg_name)) / "resume_state.pt").is_file():
                    targs.append("--resume")
                elif a.overwrite:
                    targs.append("--overwrite")
                train.main(targs)
    for m in a.models:
        cfg = load_config(f"{m}.yaml")
        print_result(evaluate(cfg, "val", a.device, force=True, batch_size=a.batch_size))
        try:
            print_result(evaluate(cfg, "test", a.device, force=False, batch_size=a.batch_size))
        except SystemExit as exc:  # e.g. test already evaluated for this frozen checkpoint
            print(f"[{m}] test evaluation skipped: {exc}")
    sys.argv = [sys.argv[0]]
    compare.main()


if __name__ == "__main__":
    main()
