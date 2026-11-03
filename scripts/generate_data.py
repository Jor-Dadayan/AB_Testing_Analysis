"""Генерация датасета: python scripts/generate_data.py [--scenario ...] [--force]."""
import argparse
import logging
import sys

import pandas as pd

from experimentlab.config import PROJECT_ROOT, load_config
from experimentlab.data.generator import run_generation
from experimentlab.data.quality import print_report, run_quality_checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["positive", "no_effect", "negative"])
    ap.add_argument("--n-users", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true", help="пересоздать данные")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    if args.scenario:
        cfg["scenario"] = args.scenario
    if args.n_users:
        cfg["n_users"] = args.n_users
    if args.seed:
        cfg["seed"] = args.seed

    csv_path = PROJECT_ROOT / cfg["paths"]["raw_csv"]
    truth_path = PROJECT_ROOT / cfg["paths"]["truth"]
    status = run_generation(cfg, csv_path, truth_path, force=args.force)
    print(f"[{status}] {csv_path}")

    df = pd.read_csv(csv_path)
    return 0 if print_report(run_quality_checks(df, cfg)) else 1


if __name__ == "__main__":
    sys.exit(main())
