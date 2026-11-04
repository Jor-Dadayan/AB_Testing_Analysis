"""Расширенный анализ: python scripts/run_advanced.py -> Markdown-отчёт и JSON."""
import argparse
import json
import logging
import sys
from pathlib import Path

from experimentlab.analysis.advanced_engine import run_advanced
from experimentlab.analysis.advanced_report import render_advanced_report
from experimentlab.analysis.eda import load_analytics_frame
from experimentlab.config import PROJECT_ROOT, load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "advanced_report.md"))
    ap.add_argument("--json", default=str(PROJECT_ROOT / "reports" / "advanced_results.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    df, source = load_analytics_frame()
    res = run_advanced(df, cfg, source)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render_advanced_report(res), encoding="utf-8")
    Path(args.json).write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Источник: {source}")
    print(f"SRM: {'пройдена' if res.srm.passed else 'НЕ ПРОЙДЕНА'} (p={res.srm.p_value:.4f})")
    c = res.base.conversion
    print(f"Конверсия: {c.diff * 100:+.2f} п.п. [{c.ci_low * 100:+.2f}; {c.ci_high * 100:+.2f}], "
          f"p={c.p_value:.2e}, вывод: {res.base.conversion_verdict.practical}")
    print("Guardrail:", ", ".join(f"{g.name}={g.status}" for g in res.guardrails))
    s = res.segments
    print(f"Сегменты: тестов {s.n_tests}; значимо сырых {int(s.table['sig_raw'].sum())}, "
          f"Holm {int(s.table['sig_holm'].sum())}, BH {int(s.table['sig_bh'].sum())}")
    print(f"Устойчивость: конверсия={res.sensitivity.conversion_robust}, ARPU={res.sensitivity.arpu_robust}")
    print(f"\nИТОГОВОЕ РЕШЕНИЕ: {res.decision.code}")
    print(f"Отчёт: {args.report}\nJSON:  {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
