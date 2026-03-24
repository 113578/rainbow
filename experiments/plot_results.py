"""
Построение графиков результатов экспериментов.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_metrics(results_dir: str) -> dict | None:
    """Загрузка метрик из файла."""
    metrics_path = os.path.join(results_dir, "metrics.pth")

    if not os.path.exists(metrics_path):
        print(f"Предупреждение: {metrics_path} не найден, пропускаем.")
        return None

    return torch.load(metrics_path, weights_only=False)


def plot_learning_curve(ax, metrics: dict, label: str, color: str | None = None):
    """Построение кривой обучения со сглаживанием скользящим средним."""
    steps = metrics["steps"]
    rewards = metrics["rewards"]

    if not steps:
        return

    mean_rewards = [np.mean(r) for r in rewards]

    # Сглаживание скользящим средним (окно=5)
    window = min(5, len(mean_rewards))

    if window > 1:
        smoothed = np.convolve(mean_rewards, np.ones(window) / window, mode="valid")
        steps_smoothed = steps[window - 1 :]
    else:
        smoothed = mean_rewards
        steps_smoothed = steps

    ax.plot(steps_smoothed, smoothed, label=label, color=color, linewidth=1.5)


def main():
    base_dir = os.path.join("experiments", "results")

    experiments = {
        "rainbow_pong": ("Rainbow (Pong)", "#1f77b4"),
        "rainbow_breakout": ("Rainbow (Breakout)", "#ff7f0e"),
        "no_priority_breakout": ("Без приоритизации (Breakout)", "#d62728"),
        "no_multistep_breakout": ("Без многошагового (Breakout)", "#9467bd"),
        "no_distributional_breakout": ("Без распределения (Breakout)", "#2ca02c"),
    }

    found = {}

    for exp_id, (label, color) in experiments.items():
        metrics = load_metrics(os.path.join(base_dir, exp_id))
        if metrics is not None:
            found[exp_id] = (label, color, metrics)

    if not found:
        print("Результаты экспериментов не найдены. Сначала запустите эксперименты.")
        sys.exit(1)

    # График 1: Кривые обучения всех экспериментов
    fig, ax = plt.subplots(figsize=(10, 6))

    for _exp_id, (label, color, metrics) in found.items():
        plot_learning_curve(ax, metrics, label, color)

    ax.set_xlabel("Шаги обучения")
    ax.set_ylabel("Средняя награда")
    ax.set_title("Rainbow DQN — Кривые обучения")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = os.path.join(base_dir, "learning_curves.png")
    fig.savefig(output_path, dpi=150)
    print(f"Сохранено: {output_path}")

    # График 2: Сравнение абляций только на Pong
    pong_experiments = {k: v for k, v in found.items() if k.endswith("_pong")}

    if len(pong_experiments) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        for _exp_id, (label, color, metrics) in pong_experiments.items():
            plot_learning_curve(ax, metrics, label, color)

        ax.set_xlabel("Шаги обучения")
        ax.set_ylabel("Средняя награда")
        ax.set_title("Rainbow DQN — Абляционное исследование на Pong")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        output_path = os.path.join(base_dir, "ablation_pong.png")
        fig.savefig(output_path, dpi=150)
        print(f"Сохранено: {output_path}")

    plt.close("all")


if __name__ == "__main__":
    main()
