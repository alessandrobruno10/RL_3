from __future__ import annotations

from typing import Dict, List, Tuple

from .gridworld import Action, GridWorld, State


class DynamicProgrammingDemo:
    """Provide simple dynamic programming algorithms for the example gridworld."""

    def __init__(self, env: GridWorld) -> None:
        self.env = env

    def policy_evaluation(self, policy: Dict[State, Action], values: Dict[State, float], theta: float = 1e-9, max_iterations: int = 1000) -> Dict[State, float]:
        current_values = dict(values)
        for _ in range(max_iterations):
            delta = 0.0
            new_values = dict(current_values)
            for state in self.env.states:
                if self.env.is_terminal(state):
                    new_values[state] = 0.0
                    continue
                action = policy[state]
                next_state, reward = self.env.step(state, action)
                updated_value = reward + self.env.gamma * current_values[next_state]
                delta = max(delta, abs(updated_value - current_values[state]))
                new_values[state] = updated_value
            current_values = new_values
            if delta < theta:
                break
        return current_values

    def policy_improvement(self, values: Dict[State, float]) -> Dict[State, Action]:
        policy: Dict[State, Action] = {}
        for state in self.env.states:
            if self.env.is_terminal(state):
                continue
            best_action = self._best_action_for_state(state, values)
            policy[state] = best_action
        return policy

    def _best_action_for_state(self, state: State, values: Dict[State, float]) -> Action:
        best_action = self.env.actions[0]
        best_value = float("-inf")
        for action in self.env.actions:
            next_state, reward = self.env.step(state, action)
            action_value = reward + self.env.gamma * values[next_state]
            if action_value > best_value or (action_value == best_value and action == "right" and best_action != "right"):
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self, theta: float = 1e-9, max_iterations: int = 1000) -> Tuple[Dict[State, float], Dict[State, Action]]:
        values: Dict[State, float] = {state: 0.0 for state in self.env.states}
        for _ in range(max_iterations):
            delta = 0.0
            new_values = dict(values)
            for state in self.env.states:
                if self.env.is_terminal(state):
                    new_values[state] = 0.0
                    continue
                best_value = max(
                    reward + self.env.gamma * values[next_state]
                    for action in self.env.actions
                    for next_state, reward in [self.env.step(state, action)]
                )
                new_values[state] = best_value
                delta = max(delta, abs(new_values[state] - values[state]))
            values = new_values
            if delta < theta:
                break
        policy = self.policy_improvement(values)
        return values, policy
