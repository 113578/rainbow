"""
Обучение Rainbow DQN.
"""

import os
from datetime import datetime
from typing import Annotated

import numpy as np
import torch
import typer
from tqdm import trange

from rainbow.agent import Agent
from rainbow.env import Env
from rainbow.memory import ReplayMemory
from rainbow.test import test

app = typer.Typer(help="Rainbow DQN - обучение и эксперименты")


def log(s: str):
    print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] {s}")


def select_device(*, disable_cuda: bool) -> torch.device:
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(np.random.randint(1, 10000))
    elif not disable_cuda and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log(f"Устройство: {device}")

    return device


@app.command()
def train(
    # Среда
    game: Annotated[
        str, typer.Option(help="Название игры Atari (например Pong, Breakout)")
    ] = "Pong",
    seed: Annotated[int, typer.Option(help="Случайное зерно")] = 123,
    max_episode_length: Annotated[
        int, typer.Option(help="Макс. кадров на эпизод")
    ] = 108_000,
    # Обучение
    id: Annotated[str, typer.Option(help="Идентификатор эксперимента")] = "default",
    t_max: Annotated[
        int, typer.Option("--t-max", help="Число шагов обучения")
    ] = 500_000,
    learn_start: Annotated[int, typer.Option(help="Шагов до начала обучения")] = 20_000,
    replay_frequency: Annotated[
        int, typer.Option(help="Частота выборки из памяти")
    ] = 4,
    batch_size: Annotated[int, typer.Option(help="Размер батча")] = 32,
    learning_rate: Annotated[
        float, typer.Option(help="Скорость обучения Adam")
    ] = 0.0000625,
    adam_eps: Annotated[float, typer.Option(help="Эпсилон Adam")] = 1.5e-4,
    discount: Annotated[float, typer.Option(help="Коэффициент дисконтирования")] = 0.99,
    norm_clip: Annotated[float, typer.Option(help="Макс. L2-норма градиента")] = 10.0,
    reward_clip: Annotated[
        float, typer.Option(help="Обрезка награды (0 = выкл.)")
    ] = 1.0,
    target_update: Annotated[
        int, typer.Option(help="Период обновления целевой сети")
    ] = 8000,
    # Архитектура
    history_length: Annotated[int, typer.Option(help="Число сложенных кадров")] = 4,
    hidden_size: Annotated[int, typer.Option(help="Размер скрытого слоя")] = 512,
    noisy_std: Annotated[float, typer.Option(help="Начальная сигма NoisyLinear")] = 0.5,
    atoms: Annotated[int, typer.Option(help="Число атомов распределения")] = 51,
    v_min: Annotated[
        float, typer.Option("--v-min", help="Минимум носителя распределения")
    ] = -10.0,
    v_max: Annotated[
        float, typer.Option("--v-max", help="Максимум носителя распределения")
    ] = 10.0,
    # Память
    memory_capacity: Annotated[
        int, typer.Option(help="Ёмкость буфера воспроизведения")
    ] = 100_000,
    priority_exponent: Annotated[
        float, typer.Option(help="Экспонента приоритета (омега)")
    ] = 0.5,
    priority_weight: Annotated[
        float, typer.Option(help="Начальный вес важности (бета)")
    ] = 0.4,
    multi_step: Annotated[int, typer.Option(help="Число шагов n-step наград")] = 3,
    # Оценка
    evaluation_interval: Annotated[
        int, typer.Option(help="Интервал оценки (шаги)")
    ] = 10_000,
    evaluation_episodes: Annotated[int, typer.Option(help="Эпизодов оценки")] = 10,
    evaluation_size: Annotated[
        int, typer.Option(help="Переходов для валидации Q")
    ] = 500,
    # Эксперименты
    no_double: Annotated[
        bool, typer.Option(help="Отключить double Q-learning")
    ] = False,
    no_priority: Annotated[bool, typer.Option(help="Отключить приоритизацию")] = False,
    no_dueling: Annotated[
        bool, typer.Option(help="Отключить дуэльную архитектуру")
    ] = False,
    no_distributional: Annotated[
        bool, typer.Option(help="Отключить распределённое RL")
    ] = False,
    no_noisy: Annotated[
        bool, typer.Option(help="Отключить NoisyNets (вместо них e-greedy)")
    ] = False,
    # Устройство
    disable_cuda: Annotated[bool, typer.Option(help="Отключить CUDA/MPS")] = False,
    # Каталог результатов
    results_dir: Annotated[
        str, typer.Option(help="Каталог для сохранения результатов")
    ] = "results",
):
    """Запуск обучения агента Rainbow DQN."""
    # Инициализация
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.benchmark = True
    device = select_device(disable_cuda=disable_cuda)

    output_dir = os.path.join(results_dir, id)
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "steps": [],
        "rewards": [],
        "Qs": [],
        "best_avg_reward": -float("inf"),
    }

    # Лог конфигурации
    log(f"Эксперимент: {id}")
    log(f"Игра: {game}")

    ablations = []

    if no_double:
        ablations.append("без double")
    if no_priority:
        ablations.append("без приоритизации")
    if no_dueling:
        ablations.append("без дуэльной сети")
    if no_distributional:
        ablations.append("без распределения")
    if no_noisy:
        ablations.append("без шумовых сетей")
    if multi_step == 1:
        ablations.append("без многошагового (n=1)")
    if ablations:
        log(f"Абляции: {', '.join(ablations)}")
    else:
        log("Полный Rainbow (без абляций)")

    # Среда
    env = Env(game, seed, device, max_episode_length, history_length)
    env.train()
    action_space = env.action_space()

    # Агент
    use_noisy = not no_noisy
    use_distributional = not no_distributional
    dqn = Agent(
        action_space,
        atoms=atoms,
        v_min=v_min,
        v_max=v_max,
        batch_size=batch_size,
        multi_step=multi_step,
        discount=discount,
        norm_clip=norm_clip,
        learning_rate=learning_rate,
        adam_eps=adam_eps,
        hidden_size=hidden_size,
        noisy_std=noisy_std,
        history_length=history_length,
        device=device,
        noisy=use_noisy,
        dueling=not no_dueling,
        double=not no_double,
        distributional=use_distributional,
    )

    # Буфер воспроизведения
    mem = ReplayMemory(
        capacity=memory_capacity,
        history=history_length,
        discount=discount,
        n=multi_step,
        priority_weight=priority_weight,
        priority_exponent=priority_exponent if not no_priority else 0.0,
        device=device,
    )

    priority_weight_increase = (1 - priority_weight) / (t_max - learn_start)

    # Валидационная память
    val_mem = ReplayMemory(
        capacity=evaluation_size,
        history=history_length,
        discount=discount,
        n=multi_step,
        priority_weight=priority_weight,
        priority_exponent=priority_exponent,
        device=device,
    )

    t_val, done = 0, True

    while t_val < evaluation_size:
        if done:
            state = env.reset()

        next_state, _, done = env.step(np.random.randint(0, action_space))
        val_mem.append(state, -1, 0.0, done)
        state = next_state
        t_val += 1

    # Эпсилон для исследования без шумовых сетей
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 250_000

    # Цикл обучения
    dqn.train()
    done = True

    for t in trange(1, t_max + 1):
        if done:
            state = env.reset()

        if use_noisy and t % replay_frequency == 0:
            dqn.reset_noise()

        # Выбор действия
        if use_noisy:
            action = dqn.act(state)
        else:
            epsilon = max(
                epsilon_final,
                epsilon_start - (epsilon_start - epsilon_final) * t / epsilon_decay,
            )
            action = dqn.act_e_greedy(state, epsilon)

        next_state, reward, done = env.step(action)

        if reward_clip > 0:
            reward = max(min(reward, reward_clip), -reward_clip)

        mem.append(state, action, reward, done)

        if t >= learn_start:
            if not no_priority:
                mem.priority_weight = min(
                    mem.priority_weight + priority_weight_increase, 1.0
                )

            if t % replay_frequency == 0:
                dqn.learn(mem)

            if t % evaluation_interval == 0:
                dqn.eval()
                avg_reward, avg_q = test(
                    game,
                    seed,
                    device,
                    t,
                    dqn,
                    val_mem,
                    metrics,
                    output_dir,
                    evaluation_episodes,
                )

                log(
                    f"T = {t} / {t_max} | "
                    f"Ср. награда: {avg_reward:.1f} | Ср. Q: {avg_q:.3f}"
                )

                dqn.train()

            if t % target_update == 0:
                dqn.update_target_net()

        state = next_state

    env.close()
    log("Обучение завершено.")


if __name__ == "__main__":
    app()
