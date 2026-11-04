"""Статистический анализ: python scripts/run_analysis.py -> Markdown-отчёт и JSON."""

import argparse
import json
import logging
import sys
from pathlib import Path

from experimentlab.analysis.eda import load_analytics_frame
from experimentlab.analysis.engine import run_analysis
from experimentlab.analysis.report import render_report
from experimentlab.config import PROJECT_ROOT, load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "statistical_report.md"))
    ap.add_argument("--json", default=str(PROJECT_ROOT / "reports" / "analysis_results.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    df, source = load_analytics_frame()
    result = run_analysis(df, cfg, source)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render_report(result), encoding="utf-8")
    Path(args.json).write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    c, w = result.conversion, result.arpu_welch
    print(f"Источник: {source}")
    print(f"SRM: {'пройдена' if result.srm.passed else 'НЕ ПРОЙДЕНА'} (p={result.srm.p_value:.4f})")
    print(
        f"Конверсия: control {c.p_control:.2%}, treatment {c.p_treatment:.2%}, "
        f"разница {c.diff * 100:+.2f} п.п. "
        f"[{c.ci_low * 100:+.2f}; {c.ci_high * 100:+.2f}], z={c.z:.2f}, p={c.p_value:.2e}"
    )
    print(f"ARPU: разница {w.diff:+.4f} [{w.ci_low:+.4f}; {w.ci_high:+.4f}], p={w.p_value:.4f}")
    print(f"Решение: {result.recommendation}")
    print(f"\nОтчёт: {args.report}\nJSON:  {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
