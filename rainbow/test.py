"""Утилиты оценки для Rainbow DQN."""

import torch

from rainbow.env import Env


def test(
    game: str,
    seed: int,
    device: torch.device,
    t: int,
    dqn,
    val_mem,
    metrics: dict,
    results_dir: str,
    evaluation_episodes: int = 10,
    *,
    evaluate: bool = False,
) -> tuple[float, float]:
    """
    Оценка производительности агента на нескольких эпизодах.

    Запускает ``evaluation_episodes`` эпизодов с ε-жадной стратегией (ε=0.001),
    вычисляет среднее Q-значение по валидационной памяти и сохраняет лучшую модель.

    Parameters
    ----------
    game : str
        Название игры Atari.
    seed : int
        Случайное зерно для среды.
    device : torch.device
        Устройство torch.
    t : int
        Текущий шаг обучения.
    dqn : Agent
        Агент для оценки.
    val_mem : ReplayMemory
        Валидационная память для оценки Q-значений.
    metrics : dict
        Словарь метрик (обновляется на месте).
    results_dir : str
        Каталог для сохранения модели и метрик.
    evaluation_episodes : int, optional
        Число эпизодов оценки. По умолчанию 10.
    evaluate : bool, optional
        Если True, не сохраняет модель и метрики. По умолчанию False.

    Returns
    -------
    avg_reward : float
        Средняя награда за эпизод.
    avg_q : float
        Среднее Q-значение по валидационной памяти.
    """
    env = Env(game, seed, device)
    env.eval()
    metrics["steps"].append(t)
    t_rewards: list[float] = []
    t_qs: list[float] = []

    done = True

    for _ in range(evaluation_episodes):
        while True:
            if done:
                state = env.reset()
                reward_sum = 0.0
                done = False

            action = dqn.act_e_greedy(state)
            state, reward, done = env.step(action)
            reward_sum += reward

            if done:
                t_rewards.append(reward_sum)
                break

    env.close()

    for state in val_mem:
        t_qs.append(dqn.evaluate_q(state))

    avg_reward = sum(t_rewards) / len(t_rewards)
    avg_q = sum(t_qs) / len(t_qs)

    if not evaluate:
        if avg_reward > metrics["best_avg_reward"]:
            metrics["best_avg_reward"] = avg_reward
            dqn.save(results_dir)

        metrics["rewards"].append(t_rewards)
        metrics["Qs"].append(t_qs)
        torch.save(metrics, f"{results_dir}/metrics.pth")

    return avg_reward, avg_q
