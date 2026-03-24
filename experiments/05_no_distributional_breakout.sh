#!/bin/bash
RESULTS_DIR="experiments/results"
T_MAX=${T_MAX:-500000}

echo ">>> Эксперимент 5: Breakout без распределённого RL (T_max=$T_MAX)"

uv run python main.py \
    --game Breakout \
    --id no_distributional_breakout \
    --no-distributional \
    --t-max "$T_MAX" \
    --evaluation-interval 10000 \
    --memory-capacity 100000 \
    --results-dir "$RESULTS_DIR"

echo ">>> Эксперимент 5 завершён"
