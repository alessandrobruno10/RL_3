# Dynamic Programming in Reinforcement Learning

This project demonstrates three classic dynamic programming methods in reinforcement learning:

- Policy Evaluation
- Policy Improvement
- Value Iteration

The example uses a tiny 2x3 gridworld where the agent learns the best actions from state values.

## Setup

Create a fresh virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the demo

```bash
PYTHONPATH=src python demo.py
```

## Run the tests

```bash
PYTHONPATH=src python -m pytest -q
```
