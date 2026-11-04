#!/bin/sh
# Разовая подготовка: данные, демо-эксперименты, загрузка в PostgreSQL. Повторный запуск безопасен.
set -eu
python scripts/generate_data.py
[ -f data/experiments/demo_3.csv ] || python scripts/build_experiments.py
python scripts/load_to_db.py
