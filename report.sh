#!/bin/bash

source venv_py310/bin/activate

SRC_ROOT="./tests/test_datas"
DST_ROOT="./tests/test_datas_results"

LINE="SSBR"
GRADE="unknown"

DATES=(
  "2026-02-20"
)

BATCH_SIZE=9

python report.py \
  --src-root "$SRC_ROOT" \
  --dst-root "$DST_ROOT" \
  --line "$LINE" \
  --grade "$GRADE" \
  --dates "${DATES[@]}" \
  --batch-size "$BATCH_SIZE"