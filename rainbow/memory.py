"""
Приоритизированный воспроизведённый опыт с деревом отрезков.

Реализует пропорциональную приоритизацию (Schaul et al. 2015) с использованием
суммирующего дерева для эффективной выборки за O(log n). Хранит отдельные кадры
и восстанавливает сложенные состояния по запросу для экономии памяти.
"""

import numpy as np
import torch

Transition_dtype = np.dtype(
    [
        ("timestep", np.int32),
        ("state", np.uint8, (84, 84)),
        ("action", np.int32),
        ("reward", np.float32),
        ("nonterminal", np.bool_),
    ]
)
blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)


class SegmentTree:
    """
    Суммирующее дерево отрезков для приоритизированной выборки за O(log n).

    Parameters
    ----------
    size : int
        Максимальное число элементов (листьев дерева).
    """

    def __init__(self, size: int):
        self.index = 0
        self.size = size
        self.full = False
        self.tree_start = 2 ** (size - 1).bit_length() - 1
        self.sum_tree = np.zeros(self.tree_start + self.size, dtype=np.float32)
        self.data = np.array([blank_trans] * size, dtype=Transition_dtype)
        self.max = 1.0

    def _update_nodes(self, indices: np.ndarray):
        """Обновление внутренних узлов дерева по индексам."""
        children = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children], axis=0)

    def _propagate(self, indices: np.ndarray):
        """Рекурсивное распространение обновления к корню (батчевое)."""
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)

        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index: int):
        """Распространение обновления одного листа к корню."""
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]

        if parent != 0:
            self._propagate_index(parent)

    def update(self, indices: np.ndarray, values: np.ndarray):
        """
        Обновление приоритетов для батча индексов.

        Parameters
        ----------
        indices : np.ndarray
            Индексы листьев в дереве.
        values : np.ndarray
            Новые значения приоритетов.
        """
        self.sum_tree[indices] = values
        self._propagate(indices)
        current_max = np.max(values)
        self.max = max(current_max, self.max)

    def _update_index(self, index: int, value: float):
        """Обновление приоритета одного листа."""
        self.sum_tree[index] = value
        self._propagate_index(index)
        self.max = max(value, self.max)

    def append(self, data: tuple, value: float):
        """
        Добавление нового перехода в циклический буфер.

        Parameters
        ----------
        data : tuple
            Кортеж перехода ``(timestep, state, action, reward, nonterminal)``.
        value : float
            Начальный приоритет перехода.
        """
        self.data[self.index] = data
        self._update_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def _retrieve(self, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Рекурсивный поиск листьев по кумулятивным значениям."""
        children = indices * 2 + np.expand_dims([1, 2], axis=1)

        if children[0, 0] >= self.sum_tree.shape[0]:
            return indices

        if children[0, 0] >= self.tree_start:
            children = np.minimum(children, self.sum_tree.shape[0] - 1)

        left_values = self.sum_tree[children[0]]
        go_right = np.greater(values, left_values).astype(np.int32)

        successor_indices = children[go_right, np.arange(indices.size)]
        successor_values = values - go_right * left_values

        return self._retrieve(successor_indices, successor_values)

    def find(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Поиск переходов по кумулятивным значениям приоритетов.

        Parameters
        ----------
        values : np.ndarray
            Кумулятивные значения для стратифицированной выборки.

        Returns
        -------
        probs : np.ndarray
            Приоритеты найденных переходов.
        data_index : np.ndarray
            Индексы переходов в массиве данных.
        tree_index : np.ndarray
            Индексы листьев в дереве.
        """
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start

        return self.sum_tree[indices], data_index, indices

    def get(self, data_index: np.ndarray):
        """
        Извлечение переходов по индексам данных.

        Parameters
        ----------
        data_index : np.ndarray
            Индексы (поддерживает циклическую адресацию).

        Returns
        -------
        np.ndarray
            Массив переходов структурированного типа.
        """
        return self.data[data_index % self.size]

    def total(self) -> float:
        """
        Полная сумма приоритетов.

        Returns
        -------
        float
            Значение корня дерева.
        """
        return float(self.sum_tree[0])


class ReplayMemory:
    """
    Буфер приоритизированного воспроизведённого опыта.

    Хранит отдельные кадры и восстанавливает сложенные состояния по запросу.
    Использует дерево отрезков для эффективной пропорциональной приоритизации.

    Parameters
    ----------
    capacity : int
        Максимальное число хранимых переходов.
    history : int
        Число кадров для стекирования состояния.
    discount : float
        Коэффициент дисконтирования для n-шаговых наград.
    n : int
        Число шагов для многошаговых наград.
    priority_weight : float
        Начальный вес выборки по важности (β).
    priority_exponent : float
        Экспонента приоритета (ω).
    device : torch.device
        Устройство torch для возвращаемых тензоров.
    """

    def __init__(
        self,
        capacity: int,
        history: int,
        discount: float,
        n: int,
        priority_weight: float,
        priority_exponent: float,
        device: torch.device,
    ):
        self.device = device
        self.capacity = capacity
        self.history = history
        self.discount = discount
        self.n = n
        self.priority_weight = priority_weight
        self.priority_exponent = priority_exponent
        self.t = 0
        self.n_step_scaling = torch.tensor(
            [self.discount**i for i in range(self.n)],
            dtype=torch.float32,
            device=self.device,
        )
        self.transitions = SegmentTree(capacity)

    def append(self, state: torch.Tensor, action: int, reward: float, terminal: bool):
        """
        Добавление перехода в буфер.

        Сохраняет только последний кадр состояния (uint8) для экономии памяти.
        Новый переход получает максимальный приоритет.

        Parameters
        ----------
        state : torch.Tensor
            Сложенное состояние формы ``(history, 84, 84)``.
        action : int
            Индекс действия.
        reward : float
            Награда.
        terminal : bool
            Признак завершения эпизода.
        """
        state_frame = (
            state[-1].mul(255).to(dtype=torch.uint8, device=torch.device("cpu"))
        )
        self.transitions.append(
            (self.t, state_frame, action, reward, not terminal),
            self.transitions.max,
        )
        self.t = 0 if terminal else self.t + 1

    def _get_transitions(self, idxs: np.ndarray):
        """
        Извлечение окон переходов с маскированием границ эпизодов.

        Parameters
        ----------
        idxs : np.ndarray
            Индексы центральных переходов.

        Returns
        -------
        np.ndarray
            Массив переходов формы ``(batch, history + n)``.
        """
        transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(
            idxs, axis=1
        )
        transitions = self.transitions.get(transition_idxs)
        transitions_firsts = transitions["timestep"] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)

        for t in range(self.history - 2, -1, -1):
            blank_mask[:, t] = np.logical_or(
                blank_mask[:, t + 1], transitions_firsts[:, t + 1]
            )

        for t in range(self.history, self.history + self.n):
            blank_mask[:, t] = np.logical_or(
                blank_mask[:, t - 1], transitions_firsts[:, t]
            )

        transitions[blank_mask] = blank_trans

        return transitions

    def _get_samples_from_segments(self, batch_size: int, p_total: float):
        """
        Стратифицированная выборка из дерева приоритетов.

        Parameters
        ----------
        batch_size : int
            Размер батча.
        p_total : float
            Полная сумма приоритетов.

        Returns
        -------
        tuple
            Кортеж ``(probs, idxs, tree_idxs, states, actions,
            n_step_return, next_states, nonterminals)``.
        """
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        valid = False

        while not valid:
            samples = (
                np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
            )
            probs, idxs, tree_idxs = self.transitions.find(samples)

            if (
                np.all((self.transitions.index - idxs) % self.capacity > self.n)
                and np.all(
                    (idxs - self.transitions.index) % self.capacity >= self.history
                )
                and np.all(probs != 0)
            ):
                valid = True

        transitions = self._get_transitions(idxs)
        all_states = transitions["state"]
        states = torch.tensor(
            all_states[:, : self.history], device=self.device, dtype=torch.float32
        ).div_(255)
        next_states = torch.tensor(
            all_states[:, self.n : self.n + self.history],
            device=self.device,
            dtype=torch.float32,
        ).div_(255)
        actions = torch.tensor(
            np.copy(transitions["action"][:, self.history - 1]),
            dtype=torch.int64,
            device=self.device,
        )
        rewards = torch.tensor(
            np.copy(transitions["reward"][:, self.history - 1 : -1]),
            dtype=torch.float32,
            device=self.device,
        )
        n_step_return = torch.matmul(rewards, self.n_step_scaling)
        nonterminals = torch.tensor(
            np.expand_dims(
                transitions["nonterminal"][:, self.history + self.n - 1], axis=1
            ),
            dtype=torch.float32,
            device=self.device,
        )

        return (
            probs,
            idxs,
            tree_idxs,
            states,
            actions,
            n_step_return,
            next_states,
            nonterminals,
        )

    def sample(
        self, batch_size: int
    ) -> tuple[
        np.ndarray,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Выборка батча с коррекцией весов важности.

        Parameters
        ----------
        batch_size : int
            Размер батча.

        Returns
        -------
        tree_idxs : np.ndarray
            Индексы листьев дерева для обновления приоритетов.
        states : torch.Tensor
            Состояния формы ``(batch, history, 84, 84)``.
        actions : torch.Tensor
            Действия формы ``(batch,)``.
        returns : torch.Tensor
            n-шаговые награды формы ``(batch,)``.
        next_states : torch.Tensor
            Следующие состояния формы ``(batch, history, 84, 84)``.
        nonterminals : torch.Tensor
            Маски нетерминальности формы ``(batch, 1)``.
        weights : torch.Tensor
            Нормализованные веса важности формы ``(batch,)``.
        """
        p_total = self.transitions.total()
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = (
            self._get_samples_from_segments(batch_size, p_total)
        )
        probs = probs / p_total
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight
        weights = torch.tensor(
            weights / weights.max(), dtype=torch.float32, device=self.device
        )

        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray):
        """
        Обновление приоритетов после обучения.

        Parameters
        ----------
        idxs : np.ndarray
            Индексы листьев дерева.
        priorities : np.ndarray
            Новые приоритеты (до возведения в степень ω).
        """
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    def __iter__(self):
        """Итератор по состояниям буфера (для оценки Q-значений)."""
        self.current_idx = 0

        return self

    def __next__(self) -> torch.Tensor:
        """
        Следующее сложенное состояние из буфера.

        Returns
        -------
        torch.Tensor
            Состояние формы ``(history, 84, 84)`` с нормализацией в [0, 1].

        Raises
        ------
        StopIteration
            Когда все переходы перебраны.
        """
        if self.current_idx == self.capacity:
            raise StopIteration

        transitions = self.transitions.data[
            np.arange(self.current_idx - self.history + 1, self.current_idx + 1)
        ]
        transitions_firsts = transitions["timestep"] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)

        for t in reversed(range(self.history - 1)):
            blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1])

        transitions[blank_mask] = blank_trans
        state = torch.tensor(
            transitions["state"], dtype=torch.float32, device=self.device
        ).div_(255)
        self.current_idx += 1

        return state
