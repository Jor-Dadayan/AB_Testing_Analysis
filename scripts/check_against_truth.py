"""Внутренняя проверка симуляции: попал ли интервал в заложенный эффект.

Скрытая правда используется только здесь и НЕ попадает в отчёты.
"""

import json

from experimentlab.analysis.eda import load_analytics_frame
from experimentlab.analysis.engine import run_analysis
from experimentlab.config import PROJECT_ROOT, load_config

cfg = load_config()
truth = json.loads((PROJECT_ROOT / cfg["paths"]["truth"]).read_text(encoding="utf-8"))
df, source = load_analytics_frame()
c = run_analysis(df, cfg, source).conversion
true_delta = truth["true_conversion_delta"]
print(f"Заложенный эффект: {true_delta * 100:+.2f} п.п.")
print(f"Оценка: {c.diff * 100:+.2f} п.п., ДИ [{c.ci_low * 100:+.2f}; {c.ci_high * 100:+.2f}]")
print("Интервал содержит истинный эффект:", c.ci_low <= true_delta <= c.ci_high)
print("(95% интервалов покрывают истину, поэтому единичный промах не означает ошибку кода.)")
