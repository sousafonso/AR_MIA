from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class SarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}

    def select_action(self, state: StateT) -> ActionT:
        # Estratégia epsilon-greedy:
        # Com probabilidade epsilon, escolhe ação aleatória (exploração).
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            # Com probabilidade 1-epsilon, aproveita o conhecimento (aproveitamento).
            # Desempate Aleatório (Tie-Breaking): para evitar bias determinístico do np.argmax,
            # filtramos todas as ações com valor máximo e escolhemos uma aleatoriamente.
            best_value = max(self.action_value_of(state, action) for action in self.actions)
            best_actions = [action for action in self.actions if self.action_value_of(state, action) == best_value]
            action = self.rng.choice(best_actions)

        # Guarda a ação na cache do agente para a atualização posterior (on-policy).
        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        bootstrap = 0.0
        # SARSA é on-policy: se a transição não for terminal, o bootstrap utiliza o valor da ação real
        # que o agente de facto escolheu para o próximo estado (next_state), lida a partir da cache.
        if not transition.done and transition.next_state is not None:
            next_action = self._selected_actions[transition.next_state]
            bootstrap = self.action_value_of(transition.next_state, next_action)

        # td_target = R_(t+1) + gamma * Q(S_(t+1), A_(t+1))
        td_target = transition.reward + self.gamma * bootstrap
        current_value = self.action_value_of(transition.state, transition.action)
        
        # Equação de Atualização do SARSA Tabular:
        # Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [td_target - Q(S_t, A_t)]
        self.Q[(transition.state, transition.action)] = current_value + self.alpha * (td_target - current_value)

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
