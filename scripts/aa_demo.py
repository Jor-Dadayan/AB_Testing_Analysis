"""A/A-демонстрация: сегментный анализ без поправок находит «эффекты», которых нет."""
import numpy as np

from experimentlab.analysis.eda import load_analytics_frame
from experimentlab.analysis.segments import analyze_segments
from experimentlab.config import load_config

cfg = load_config()
df, _ = load_analytics_frame()
control = df[df["group_name"].astype(str) == "control"].copy()  # настоящих различий здесь нет
dims = cfg["advanced"]["segment_dimensions"]
rng = np.random.default_rng(0)
runs = 100
hits = {"raw": 0, "bonferroni": 0, "holm": 0, "bh": 0}
for _ in range(runs):
    fake = control.assign(group_name=np.where(rng.random(len(control)) < 0.5, "treatment", "control"))
    table = analyze_segments(fake, dims, run_heterogeneity=False).table
    for k in hits:
        hits[k] += bool(table[f"sig_{k}"].any())
print(f"{runs} A/A-экспериментов (разница между группами НЕ заложена):")
for k, label in (("raw", "без поправки"), ("bonferroni", "Bonferroni"), ("holm", "Holm"), ("bh", "Benjamini-Hochberg")):
    print(f"  хотя бы один «значимый» сегмент, {label:<20}: {hits[k] / runs:.0%}")
