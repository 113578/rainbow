#!/bin/bash
# Запуск всех 5 экспериментов последовательно.
#
# Переменные окружения:
#   T_MAX — число шагов обучения (по умолчанию 500000)
#
# Примеры:
#   ./experiments/run_all.sh              # 500K шагов
#   T_MAX=100000 ./experiments/run_all.sh # 100K шагов (для быстрой проверки)
set -euo pipefail

echo "=== Эксперименты Rainbow DQN ==="
echo "T_MAX=${T_MAX:-500000}"
echo ""

for script in experiments/0*.sh; do
    bash "$script"
    echo ""
done

echo "=== Все эксперименты завершены ==="
echo "Построение графиков..."
uv run python experiments/plot_results.py
echo "Готово. Результаты в experiments/results/"
