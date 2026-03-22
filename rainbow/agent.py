"""
Агент Rainbow DQN.

Объединяет шесть расширений DQN:

1. Double Q-learning (van Hasselt et al. 2016).
2. Prioritized Experience Replay (Schaul et al. 2015).
3. Dueling Networks (Wang et al. 2016).
4. Multi-step RL (Sutton 1988).
5. Distributed RL (Bellemare et al. 2017).
6. Noisy Networks (Fortunato et al. 2017).

Каждый компонент может быть независимо отключён.
"""

import os

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from rainbow.model import DQN


class Agent:
    """
    Агент Rainbow DQN.

    Parameters
    ----------
    action_space : int
        Количество дискретных действий.
    atoms : int, optional
        Число атомов распределения (1 = без распределения). По умолчанию 51.
    v_min : float, optional
        Минимум носителя распределения. По умолчанию -10.0.
    v_max : float, optional
        Максимум носителя распределения. По умолчанию 10.0.
    batch_size : int, optional
        Размер батча для обучения. По умолчанию 32.
    multi_step : int, optional
        Число шагов для многошаговых наград. По умолчанию 3.
    discount : float, optional
        Коэффициент дисконтирования γ. По умолчанию 0.99.
    norm_clip : float, optional
        Макс. L2-норма для обрезки градиентов. По умолчанию 10.0.
    learning_rate : float, optional
        Шаг обучения Adam. По умолчанию 6.25e-5.
    adam_eps : float, optional
        Эпсилон Adam. По умолчанию 1.5e-4.
    hidden_size : int, optional
        Размер скрытого слоя. По умолчанию 512.
    noisy_std : float, optional
        Начальное стандартное отклонение NoisyLinear (σ₀). По умолчанию 0.5.
    history_length : int, optional
        Число сложенных кадров. По умолчанию 4.
    device : torch.device, optional
        Устройство torch. По умолчанию CPU.
    noisy : bool, optional
        Использовать NoisyLinear слои. По умолчанию True.
    dueling : bool, optional
        Использовать дуэльную архитектуру. По умолчанию True.
    double : bool, optional
        Использовать двойное Q-обучение. По умолчанию True.
    distributional : bool, optional
        Использовать распределённое RL. По умолчанию True.
    """

    def __init__(
        self,
        action_space: int,
        *,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        batch_size: int = 32,
        multi_step: int = 3,
        discount: float = 0.99,
        norm_clip: float = 10.0,
        learning_rate: float = 0.0000625,
        adam_eps: float = 1.5e-4,
        hidden_size: int = 512,
        noisy_std: float = 0.5,
        history_length: int = 4,
        device: torch.device = torch.device("cpu"),
        noisy: bool = True,
        dueling: bool = True,
        double: bool = True,
        distributional: bool = True,
    ):
        self.action_space = action_space
        self.batch_size = batch_size
        self.n = multi_step
        self.discount = discount
        self.norm_clip = norm_clip
        self.device = device
        self.double = double
        self.distributional = distributional
        self.noisy = noisy

        if distributional:
            self.atoms = atoms
            self.v_min = v_min
            self.v_max = v_max
            self.support = torch.linspace(v_min, v_max, atoms).to(device=device)
            self.delta_z = (v_max - v_min) / (atoms - 1)
        else:
            self.atoms = 1
            self.support = torch.zeros(1, device=device)

        net_atoms = atoms if distributional else 1

        self.online_net = DQN(
            action_space,
            atoms=net_atoms,
            hidden_size=hidden_size,
            noisy_std=noisy_std,
            history_length=history_length,
            noisy=noisy,
            dueling=dueling,
        ).to(device=device)

        self.online_net.train()

        self.target_net = DQN(
            action_space,
            atoms=net_atoms,
            hidden_size=hidden_size,
            noisy_std=noisy_std,
            history_length=history_length,
            noisy=noisy,
            dueling=dueling,
        ).to(device=device)

        self.update_target_net()
        self.target_net.train()

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(
            self.online_net.parameters(), lr=learning_rate, eps=adam_eps
        )

    def reset_noise(self):
        """Перегенерация шума в онлайн-сети."""
        self.online_net.reset_noise()

    def act(self, state: torch.Tensor) -> int:
        """
        Выбор действия по жадной стратегии.

        Parameters
        ----------
        state : torch.Tensor
            Состояние формы ``(history, 84, 84)``.

        Returns
        -------
        int
            Индекс действия с максимальным Q-значением.
        """
        with torch.no_grad():
            if self.distributional:
                q = (self.online_net(state.unsqueeze(0)) * self.support).sum(2)
            else:
                q = self.online_net(state.unsqueeze(0))

            return q.argmax(1).item()

    def act_e_greedy(self, state: torch.Tensor, epsilon: float = 0.001) -> int:
        """
        Выбор действия по ε-жадной стратегии.

        Parameters
        ----------
        state : torch.Tensor
            Состояние формы ``(history, 84, 84)``.
        epsilon : float, optional
            Вероятность случайного действия. По умолчанию 0.001.

        Returns
        -------
        int
            Индекс выбранного действия.
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_space)

        return self.act(state)

    def learn(self, mem) -> None:
        """
        Один шаг обучения на батче из буфера воспроизведения.

        Parameters
        ----------
        mem : ReplayMemory
            Буфер приоритизированного воспроизведённого опыта.
        """
        if self.distributional:
            self._learn_distributional(mem)
        else:
            self._learn_dqn(mem)

    def _learn_distributional(self, mem) -> None:
        """
        Обучение с распределённой функцией потерь.

        Вычисляет кросс-энтропию между проекцией Беллмана целевого распределения
        и предсказанным распределением онлайн-сети.

        Parameters
        ----------
        mem : ReplayMemory
            Буфер приоритизированного воспроизведённого опыта.
        """
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(
            self.batch_size
        )

        # Лог-вероятности текущего состояния
        log_ps = self.online_net(states, log=True)
        log_ps_a = log_ps[range(self.batch_size), actions]

        with torch.no_grad():
            # Новый шум для выбора действия в следующем состоянии
            self.online_net.reset_noise()
            pns = self.online_net(next_states)
            dns = self.support.expand_as(pns) * pns

            if self.double:
                argmax_ns = dns.sum(2).argmax(1)
            else:
                pns_target = self.target_net(next_states)
                dns_target = self.support.expand_as(pns_target) * pns_target
                argmax_ns = dns_target.sum(2).argmax(1)

            # Новый шум для оценки целевой сетью
            self.target_net.reset_noise()
            pns = self.target_net(next_states)
            pns_a = pns[range(self.batch_size), argmax_ns]

            # Проекция Беллмана
            tz = returns.unsqueeze(1) + nonterminals * (
                self.discount**self.n
            ) * self.support.unsqueeze(0)
            tz = tz.clamp(min=self.v_min, max=self.v_max)

            b = (tz - self.v_min) / self.delta_z

            lower = b.floor().to(torch.int64)
            upper = b.ceil().to(torch.int64)

            lower[(upper > 0) * (lower == upper)] -= 1
            upper[(lower < (self.atoms - 1)) * (lower == upper)] += 1

            # Распределение вероятностной массы
            m = states.new_zeros(self.batch_size, self.atoms)

            offset = (
                torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size)
                .unsqueeze(1)
                .expand(self.batch_size, self.atoms)
                .to(actions)
            )

            m.view(-1).index_add_(
                0, (lower + offset).view(-1), (pns_a * (upper.float() - b)).view(-1)
            )
            m.view(-1).index_add_(
                0, (upper + offset).view(-1), (pns_a * (b - lower.float())).view(-1)
            )

        # Кросс-энтропийная функция потерь
        loss = -torch.sum(m * log_ps_a, 1)

        self.online_net.zero_grad()

        (weights * loss).mean().backward()

        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)

        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def _learn_dqn(self, mem) -> None:
        """
        Обучение без распределения (MSE-потери на скалярных Q-значениях).

        Parameters
        ----------
        mem : ReplayMemory
            Буфер приоритизированного воспроизведённого опыта.
        """
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(
            self.batch_size
        )

        # Текущие Q-значения
        q = self.online_net(states)
        q_a = q[range(self.batch_size), actions]

        with torch.no_grad():
            # Новый шум для выбора и оценки действий
            self.online_net.reset_noise()
            self.target_net.reset_noise()

            if self.double:
                argmax_ns = self.online_net(next_states).argmax(1)
                q_target = self.target_net(next_states)[
                    range(self.batch_size), argmax_ns
                ]
            else:
                q_target = self.target_net(next_states).max(1)[0]

            target = (
                returns + nonterminals.squeeze(1) * (self.discount**self.n) * q_target
            )

        # MSE
        loss = (q_a - target) ** 2

        self.online_net.zero_grad()

        (weights * loss).mean().backward()

        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)

        self.optimiser.step()

        # Приоритет = |TD-ошибка|
        td_error = (q_a - target).detach().abs().cpu().numpy()

        mem.update_priorities(idxs, td_error)

    def update_target_net(self):
        """Копирование весов онлайн-сети в целевую сеть."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str, name: str = "model.pth"):
        """
        Сохранение весов онлайн-сети.

        Parameters
        ----------
        path : str
            Каталог для сохранения.
        name : str, optional
            Имя файла. По умолчанию ``"model.pth"``.
        """
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    def evaluate_q(self, state: torch.Tensor) -> float:
        """
        Оценка максимального Q-значения для состояния.

        Parameters
        ----------
        state : torch.Tensor
            Состояние формы ``(history, 84, 84)``.

        Returns
        -------
        float
            Максимальное Q-значение по действиям.
        """
        with torch.no_grad():
            if self.distributional:
                return (
                    (self.online_net(state.unsqueeze(0)) * self.support)
                    .sum(2)
                    .max(1)[0]
                    .item()
                )
            return self.online_net(state.unsqueeze(0)).max(1)[0].item()

    def train(self):
        """Переключение онлайн-сети в режим обучения."""
        self.online_net.train()

    def eval(self):
        """Переключение онлайн-сети в режим оценки."""
        self.online_net.eval()
