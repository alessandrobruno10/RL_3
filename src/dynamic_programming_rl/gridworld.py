from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

State = Tuple[int, int]
Action = str


class GridWorld:
    """A tiny 2x3 gridworld for illustrating dynamic programming."""

    def __init__(self) -> None:
        self.states: List[State] = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        self.terminal_states = {(0, 2)}
        self.actions: List[Action] = ["up", "down", "left", "right"]
        self.gamma = 0.9
        self.reward = -1.0
        self.transition_prob = 1.0

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

    def next_state(self, state: State, action: Action) -> State:
        if self.is_terminal(state):
            return state

        row, col = state
        if action == "up":
            next_row = max(row - 1, 0)
        elif action == "down":
            next_row = min(row + 1, 1)
        elif action == "left":
            next_col = max(col - 1, 0)
            return (row, next_col)
        elif action == "right":
            next_col = min(col + 1, 2)
            return (row, next_col)
        else:
            raise ValueError(f"Unsupported action: {action}")

        return (next_row, col)

    def step(self, state: State, action: Action) -> Tuple[State, float]:
        if self.is_terminal(state):
            return state, 0.0
        next_state = self.next_state(state, action)
        return next_state, self.reward if next_state != state else self.reward

    def available_actions(self, state: State) -> List[Action]:
        if self.is_terminal(state):
            return []
        return self.actions
