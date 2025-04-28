## Related Prior Work:

**Emergent Flocking Behavior with Multi-Agent Reinforcement Learning**  
[https://arxiv.org/abs/1905.04077](https://arxiv.org/abs/1905.04077)

This work asks whether **agents trained solely to survive** in the presence of a predator (with no explicit flocking rules or rewards) will nevertheless **emerge into flocking behavior**.  
They propose the **SELFish framework**, and show that flock‐like formations arise even in a **continuous action setting**, purely from the objective of **maximizing survival time**.

**Limitations:**
1. **Single-agent training:**  
   - Although at deployment the learned network is copied across the whole group, **during training only one learning agent interacts** with static copies of itself in the environment.  
   - Thus, there is **no true multi-agent game** or **co-learning process**.  
   - The framework cannot capture or evaluate stability, cooperation, or competition under **policy heterogeneity**.

2. **Over-simplified environment:**  
   - No obstacles, no terrain variation; agents and predator move in a simple open bounded arena.  
   - The model ignores environmental complexities (e.g., cover, bottlenecks) common in real-world swarm behavior.  
   - The reward function is purely based on **survival time**: no foraging, migration, or collective tasks are modeled, limiting it to a narrow “fleeing” dynamic.


**Collective Adaptation in Multi-Agent Systems: How Predator Confusion Shapes Swarm-Like Behaviors**  
[https://arxiv.org/abs/2209.06338](https://arxiv.org/abs/2209.06338)

This study extends the exploration of predator-driven collective behaviors, running experiments with **15 and 60 agents** and **one predator**.  
It compares two observation models:
- **GOM (Global Observation Model):** Agents can always observe the predator.
- **LOM (Local Observation Model):** Agents have limited, localized perception.

**Limitations:**
- **Shared policy across all agents:**  
  - All 15 or 60 agents share a **single trained policy**.  
  - No independent learning or adaptation; thus, observed group behavior largely reflects **policy cloning** rather than **true emergent social structure**.

- **Artificial convergence:**  
  - Using the same neural network for all agents inevitably homogenizes behavior, masking the complexities of coordination, role differentiation, or conflict seen in real multi-agent systems.


**In summary:**  
Both studies conflate **policy replication artifacts** with **genuine emergent social dynamics**.  
Their findings, while suggestive, are fundamentally limited by **single-policy architectures** and **over-simplified environments**, leaving open the challenge of demonstrating truly emergent, diverse, and adaptive group behavior in more realistic multi-agent predator-prey settings.


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


# SAC 马太效应 and DQN 
They don't see each other: 

![image](https://github.com/user-attachments/assets/f1fee172-2dc4-470c-a0e2-db422cdfeb09)

They see each other: 

