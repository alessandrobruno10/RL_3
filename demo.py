from dynamic_programming_rl.dp import DynamicProgrammingDemo
from dynamic_programming_rl.gridworld import GridWorld


def main() -> None:
    env = GridWorld()
    demo = DynamicProgrammingDemo(env)

    initial_policy = {state: "right" for state in env.states if not env.is_terminal(state)}
    initial_values = {state: 0.0 for state in env.states}

    evaluated = demo.policy_evaluation(initial_policy, initial_values)
    improved_policy = demo.policy_improvement(evaluated)
    values, final_policy = demo.value_iteration()

    print("Initial policy evaluation values:")
    for state in env.states:
        print(f"  {state}: {evaluated[state]:.3f}")

    print("\nImproved policy:")
    for state in env.states:
        if not env.is_terminal(state):
            print(f"  {state}: {improved_policy[state]}")

    print("\nValue iteration results:")
    for state in env.states:
        print(f"  {state}: value={values[state]:.3f}, action={final_policy[state] if not env.is_terminal(state) else 'terminal'}")


if __name__ == "__main__":
    main()
