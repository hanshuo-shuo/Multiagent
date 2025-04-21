# Multi-Agent Reinforcement Learning for Predator-Prey Environment

This repository implements reinforcement learning algorithms (PPO and DQN) for training intelligent agents in a predator-prey environment. Agents must reach their goals while avoiding a predator.

## Environment
- Multi-agent environment with two prey agents and one predator agent
- Prey agents must reach their goals without being caught by the predator
- Support for both discrete and continuous action spaces
- Based on CellWorld framework with PettingZoo integration

## Algorithms

### PPO (Proximal Policy Optimization)
Train agents with discrete action space:
```
python train_complete.py --epoch 200 --step-per-epoch 2048 --lr 3e-4 --save-path "./models/improved/"
```

### DQN (Deep Q-Network)
Train agents with discrete action space:
```
python train_dqn.py --epoch 30 --step-per-epoch 2000 --test-num 1 --batch-size 256 --render-test
```

## Additional Features
- Custom reward functions
- Parallel environment support for multi-agent training
- TensorBoard logging for tracking training progress