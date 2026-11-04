"""Создаёт демонстрационные эксперименты в data/experiments/ для выбора в dashboard."""

import copy

from experimentlab.config import PROJECT_ROOT, load_config
from experimentlab.data.generator import generate_users

# Сценарии выводятся только в терминале: интерфейс их не раскрывает.
DEMOS = [("demo_1", "positive", 101), ("demo_2", "no_effect", 102), ("demo_3", "negative", 103)]


def main() -> None:
    base = load_config()
    out = PROJECT_ROOT / "data" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    for exp_id, scenario, seed in DEMOS:
        cfg = copy.deepcopy(base)
        cfg["scenario"], cfg["seed"] = scenario, seed
        df, _truth = generate_users(cfg)  # скрытая правда не сохраняется
        path = out / f"{exp_id}.csv"
        df.to_csv(path, index=False)
        print(f"{exp_id}: {len(df):,} строк -> {path}  (внутренний сценарий: {scenario})")


if __name__ == "__main__":
    main()
