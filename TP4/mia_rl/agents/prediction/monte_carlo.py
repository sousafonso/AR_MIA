from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class FirstVisitMonteCarloPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def reset(self) -> None:
        self.V = defaultdict(float) #estimated value of the state
        self.N = defaultdict(int) #how many times have I already seen this state as a first visit?

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        """Update the value table using first-visit Monte Carlo prediction.

        TODO:
        1. Traverse the episode backwards to compute the return G for each time step.
        2. Identify the first visit of each state within the episode.
        3. Update the sample-average estimate using self.N[state].
            # Incremental mean update:
            # V_new = V_old + (G - V_old) / N
        """
        returns = [0.0] * len(episode.transitions)
        G = 0.0
        for index in range(len(episode.transitions) - 1, -1, -1):
            transition = episode.transitions[index]
            G = transition.reward + self.gamma * G
            returns[index] = G

        seen_states: set[BlackjackState] = set()
        for transition, G in zip(episode.transitions, returns):
            state = transition.state
            if state in seen_states:
                continue
            seen_states.add(state)
            self.N[state] += 1
            self.V[state] += (G - self.V[state]) / self.N[state]
    
    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])  #the agent’s current estimate for that state
