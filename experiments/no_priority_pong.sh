#!/bin/bash
RESULTS_DIR="experiments/results"
T_MAX=${T_MAX:-500000}

echo ">>> Эксперимент 3: Pong без приоритизации (T_max=$T_MAX)"

uv run python main.py \
    --game Pong \
    --id no_priority_pong \
    --no-priority \
    --t-max "$T_MAX" \
    --evaluation-interval 10000 \
    --memory-capacity 100000 \
    --results-dir "$RESULTS_DIR"

echo ">>> Эксперимент 3 завершён"
