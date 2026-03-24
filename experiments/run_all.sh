#!/bin/bash
set -euo pipefail

echo "=== Эксперименты Rainbow DQN ==="
echo ""

for script in experiments/0*.sh; do
    bash "$script"
    echo ""
done

echo "=== Все эксперименты завершены ==="
echo "Построение графиков..."

uv run python experiments/plot_results.py

echo "Готово"
