"""
Архитектуры нейронных сетей для Rainbow DQN.

Реализует факторизованный NoisyLinear слой (Fortunato et al. 2017)
и дуэльную DQN с распределённым выходом (Wang et al. 2016; Bellemare et al. 2017).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """
    Факторизованный линейный слой с шумом.

    Использует факторизованный гауссовский шум для уменьшения
    числа независимых переменных шума при сохранении свойств исследования.

    Parameters
    ----------
    in_features : int
        Число входных признаков.
    out_features : int
        Число выходных признаков.
    std_init : float, optional
        Начальное стандартное отклонение шума (σ₀). По умолчанию 0.5.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Инициализация параметров μ и σ согласно NoisyNets (Fortunato et al. 2017)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """
        Генерация масштабированного шума: f(x) = sign(x) * sqrt(|x|).

        Parameters
        ----------
        size : int
            Размерность вектора шума.

        Returns
        -------
        torch.Tensor
            Масштабированный вектор шума.
        """
        x = torch.randn(size, device=self.weight_mu.device)

        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Генерация нового факторизованного шума ε = f(εᵢ) ⊗ f(εⱼ)."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход.

        В режиме обучения применяет шум: y = (μ + σ⊙ε)x + (μ_b + σ_b⊙ε_b).
        В режиме оценки использует только μ.

        Parameters
        ----------
        x : torch.Tensor
            Входной тензор формы ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Выходной тензор формы ``(batch, out_features)``.
        """
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )

        return F.linear(x, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    """
    Сеть DQN с поддержкой дуэльной архитектуры, распределённого выхода и шумовых слоёв.

    Parameters
    ----------
    action_space : int
        Количество дискретных действий.
    atoms : int, optional
        Число атомов для распределённого RL (1 = без распределения). По умолчанию 51.
    hidden_size : int, optional
        Размер скрытого слоя потоков значения/преимущества. По умолчанию 512.
    noisy_std : float, optional
        Начальное стандартное отклонение для NoisyLinear слоёв. По умолчанию 0.5.
    history_length : int, optional
        Число сложенных кадров (входные каналы). По умолчанию 4.
    noisy : bool, optional
        Использовать NoisyLinear (иначе обычный ``nn.Linear``). По умолчанию True.
    dueling : bool, optional
        Использовать дуэльную архитектуру (иначе один поток). По умолчанию True.
    """

    def __init__(
        self,
        action_space: int,
        atoms: int = 51,
        hidden_size: int = 512,
        noisy_std: float = 0.5,
        history_length: int = 4,
        *,
        noisy: bool = True,
        dueling: bool = True,
    ):
        super().__init__()
        self.atoms = atoms
        self.action_space = action_space
        self.dueling = dueling

        self.convs = nn.Sequential(
            nn.Conv2d(history_length, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.conv_output_size = 3136  # 64 * 7 * 7

        linear_cls = NoisyLinear if noisy else nn.Linear
        extra = {"std_init": noisy_std} if noisy else {}

        if dueling:
            self.fc_h_v = linear_cls(self.conv_output_size, hidden_size, **extra)
            self.fc_h_a = linear_cls(self.conv_output_size, hidden_size, **extra)
            self.fc_z_v = linear_cls(hidden_size, atoms, **extra)
            self.fc_z_a = linear_cls(hidden_size, action_space * atoms, **extra)
        else:
            self.fc_h = linear_cls(self.conv_output_size, hidden_size, **extra)
            self.fc_z = linear_cls(hidden_size, action_space * atoms, **extra)

    def forward(self, x: torch.Tensor, log: bool = False) -> torch.Tensor:
        """
        Прямой проход сети.

        Parameters
        ----------
        x : torch.Tensor
            Входные кадры формы ``(batch, history, 84, 84)``.
        log : bool, optional
            Вернуть лог-вероятности вместо вероятностей. По умолчанию False.

        Returns
        -------
        torch.Tensor
            При ``atoms > 1``: распределение формы ``(batch, actions, atoms)``.
            При ``atoms == 1``: Q-значения формы ``(batch, actions)``.
        """
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)

        if self.dueling:
            v = self.fc_z_v(F.relu(self.fc_h_v(x)))
            a = self.fc_z_a(F.relu(self.fc_h_a(x)))
            v = v.view(-1, 1, self.atoms)
            a = a.view(-1, self.action_space, self.atoms)
            q = v + a - a.mean(1, keepdim=True)
        else:
            q = self.fc_z(F.relu(self.fc_h(x)))
            q = q.view(-1, self.action_space, self.atoms)

        # Без распределения: возвращаем сырые Q-значения
        if self.atoms == 1:
            return q.squeeze(-1)

        # С распределением: возвращаем лог-вероятности
        if log:
            return F.log_softmax(q, dim=2)

        return F.softmax(q, dim=2)

    def reset_noise(self):
        """Перегенерация шума во всех NoisyLinear слоях."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
