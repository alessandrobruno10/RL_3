from dynamic_programming_rl.dp import DynamicProgrammingDemo
from dynamic_programming_rl.gridworld import GridWorld


def test_policy_evaluation_updates_values():
    env = GridWorld()
    demo = DynamicProgrammingDemo(env)
    policy = {state: "right" for state in env.states if not env.is_terminal(state)}
    values = {state: 0.0 for state in env.states}

    updated = demo.policy_evaluation(policy, values)

    assert updated[(0, 0)] < 0.0
    assert updated[(0, 2)] == 0.0


def test_policy_improvement_prefers_better_actions():
    env = GridWorld()
    demo = DynamicProgrammingDemo(env)
    values = {(0, 0): -1.0, (0, 1): -1.0, (0, 2): 0.0, (1, 0): -1.0, (1, 1): -1.0, (1, 2): 0.0}

    policy = demo.policy_improvement(values)

    assert policy[(0, 0)] == "right"


def test_value_iteration_returns_policy():
    env = GridWorld()
    demo = DynamicProgrammingDemo(env)

    values, policy = demo.value_iteration()

    assert values[(0, 2)] == 0.0
    assert policy[(0, 0)] in {"up", "down", "left", "right"}
