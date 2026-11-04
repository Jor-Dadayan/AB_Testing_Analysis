#!/usr/bin/env bash
# Полная проверка качества. Вывод сохраняется в reports/qa_run.txt (его и показывайте как доказательство).
set -uo pipefail
mkdir -p reports
{
  echo "== $(date -u +%FT%TZ) | $(python --version) =="
  ruff check . ; echo "ruff check exit code: $?"
  ruff format --check src tests scripts ; echo "ruff format exit code: $?"
  REQUIRE_DB=1 pytest -ra --cov=experimentlab --cov-report=term-missing --durations=10
  echo "pytest exit code: $?"
} 2>&1 | tee reports/qa_run.txt
