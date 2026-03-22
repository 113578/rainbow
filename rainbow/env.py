"""
Обёртка среды Atari на основе Gymnasium.

Выполняет предобработку кадров (оттенки серого, масштабирование, стекирование),
повтор действий и сигнал конца при потере жизни для обучения.
"""

from collections import deque

import ale_py  # noqa: F401 — регистрирует среды ALE в Gymnasium
import gymnasium as gym
import torch
from gymnasium.wrappers import AtariPreprocessing


class Env:
    """
    Среда Atari с предобработкой в стиле DQN.

    Parameters
    ----------
    game : str
        Название игры (например, ``"Pong"``, ``"Breakout"``).
    seed : int
        Состояние случайности.
    device : torch.device
        Устройство.
    max_episode_length : int, optional
        Максимум кадров на эпизод (108 000 ≈ 30 мин при 60 fps). По умолчанию 108 000.
    history_length : int, optional
        Число кадров для стекирования состояния. По умолчанию 4.
    """

    def __init__(
        self,
        game: str,
        seed: int,
        device: torch.device,
        max_episode_length: int = 108_000,
        history_length: int = 4,
    ):
        self.device = device
        self.window = history_length
        self.training = True

        self.env = gym.make(
            f"ALE/{game}-v5",
            frameskip=1,
            repeat_action_probability=0,
            max_episode_steps=max_episode_length // 4,
        )
        self.env = AtariPreprocessing(
            self.env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            scale_obs=False,
        )

        self.seed = seed
        self._seeded = False

        self.state_buffer: deque[torch.Tensor] = deque(maxlen=history_length)
        self.lives = 0
        self.life_termination = False

    def _reset_buffer(self):
        """Заполнение буфера нулевыми кадрами."""
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, dtype=torch.uint8))

    def _get_state(self) -> torch.Tensor:
        """
        Сборка сложенного состояния из буфера.

        Returns
        -------
        torch.Tensor
            Тензор формы ``(history, 84, 84)`` с нормализацией в [0, 1].
        """
        return (
            torch.stack(list(self.state_buffer), 0)
            .to(dtype=torch.float32, device=self.device)
            .div_(255)
        )

    def reset(self) -> torch.Tensor:
        """
        Сброс среды.

        При потере жизни (``life_termination``) выполняет no-op действие
        вместо полного сброса, сохраняя состояние игры.

        Returns
        -------
        torch.Tensor
            Начальное состояние формы ``(history, 84, 84)``.
        """
        if self.life_termination:
            self.life_termination = False
            obs, _, terminated, truncated, info = self.env.step(0)

            if terminated or truncated:
                obs, info = self.env.reset()
                self._reset_buffer()

        else:
            seed = self.seed if not self._seeded else None
            obs, info = self.env.reset(seed=seed)
            self._seeded = True
            self._reset_buffer()

        self.state_buffer.append(torch.tensor(obs, dtype=torch.uint8))
        self.lives = info.get("lives", 0)

        return self._get_state()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        """
        Выполнение одного шага в среде.

        В режиме обучения потеря жизни при оставшихся жизнях
        интерпретируется как конец эпизода.

        Parameters
        ----------
        action : int
            Индекс действия.

        Returns
        -------
        state : torch.Tensor
            Новое состояние формы ``(history, 84, 84)``.
        reward : float
            Награда за шаг.
        done : bool
            Признак завершения эпизода.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.state_buffer.append(torch.tensor(obs, dtype=torch.uint8))

        if self.training:
            lives = info.get("lives", 0)

            if lives < self.lives and lives > 0:
                self.life_termination = not done
                done = True

            self.lives = lives

        return self._get_state(), float(reward), done

    def action_space(self) -> int:
        """
        Число доступных действий.

        Returns
        -------
        int
            Размер пространства действий.
        """
        return int(self.env.action_space.n)  # ty: ignore[unresolved-attribute]

    def train(self):
        """Переключение в режим обучения (потеря жизни = конец эпизода)."""
        self.training = True

    def eval(self):
        """Переключение в режим оценки (потеря жизни не завершает эпизод)."""
        self.training = False

    def close(self):
        """Освобождение ресурсов среды."""
        self.env.close()
