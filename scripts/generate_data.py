"""Генерация датасета: python scripts/generate_data.py [--scenario ...] [--force]."""

import argparse
import logging
import sys

import pandas as pd

from experimentlab.config import PROJECT_ROOT, load_config
from experimentlab.data.generator import run_generation
from experimentlab.data.quality import print_report, run_quality_checks
from experimentlab.logging_setup import configure_logging

log = logging.getLogger("experimentlab.generate")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше 0")
    return number


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["positive", "no_effect", "negative"])
    ap.add_argument("--n-users", type=_positive_int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true", help="пересоздать данные")
    args = ap.parse_args()

    configure_logging()
    cfg = load_config()
    if args.scenario is not None:
        cfg["scenario"] = args.scenario
    if args.n_users is not None:
        cfg["n_users"] = args.n_users
    if args.seed is not None:  # seed = 0 тоже допустим
        cfg["seed"] = args.seed

    csv_path = PROJECT_ROOT / cfg["paths"]["raw_csv"]
    truth_path = PROJECT_ROOT / cfg["paths"]["truth"]
    try:
        status = run_generation(cfg, csv_path, truth_path, force=args.force)
        print(f"[{status}] {csv_path}")
        df = pd.read_csv(csv_path)
    except (OSError, ValueError, KeyError) as exc:
        log.error("Генерация не удалась: %s", exc)
        return 1
    return 0 if print_report(run_quality_checks(df, cfg)) else 1


if __name__ == "__main__":
    sys.exit(main())
