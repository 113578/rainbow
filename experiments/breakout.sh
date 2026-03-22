#!/bin/bash
RESULTS_DIR="experiments/results"
T_MAX=${T_MAX:-500000}

echo ">>> Эксперимент 2: Breakout (T_max=$T_MAX)"

uv run python main.py \
    --game Breakout \
    --id rainbow_breakout \
    --t-max "$T_MAX" \
    --evaluation-interval 10000 \
    --memory-capacity 100000 \
    --results-dir "$RESULTS_DIR"

echo ">>> Эксперимент 2 завершён"
