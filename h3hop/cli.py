"""Thin dispatcher so `h3hop <cmd>` maps to a module main().

Each module also runs standalone via `python -m h3hop.<module>`.
"""
from __future__ import annotations

import importlib
import sys

SUBCOMMANDS = {
    "gen-good": "h3hop.gen_good",
    "gen-bad": "h3hop.gen_bad",
    "run-tests": "h3hop.filter.run_tests",
    "static-checks": "h3hop.filter.static_checks",
    "decontaminate": "h3hop.filter.decontaminate",
    "build-dataset": "h3hop.build_dataset",
    "build-pairs": "h3hop.build_pairs",
    "train-sft": "h3hop.train_sft",
    "train-gkd": "h3hop.train_gkd",
    "train-dpo": "h3hop.train_dpo",
    "train-cringe": "h3hop.train_cringe",
    "train-rlvr": "h3hop.train_rlvr",
    "build-control-vector": "h3hop.build_control_vector",
    "build-anti-expert": "h3hop.build_anti_expert",
    "merge-lora": "h3hop.merge_lora",
    "infer": "h3hop.infer_demo",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("h3hop <command> [args]\n\ncommands:")
        for k in SUBCOMMANDS:
            print(f"  {k}")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in SUBCOMMANDS:
        raise SystemExit(f"unknown command: {cmd}")
    mod = importlib.import_module(SUBCOMMANDS[cmd])
    mod.main(rest)


if __name__ == "__main__":
    main()
