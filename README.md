
## Related Prior Work

**Emergent Escape-based Flocking Behavior using Multi-Agent Reinforcement Learning**  
[https://arxiv.org/abs/1905.04077](https://arxiv.org/abs/1905.04077)

This paper asks if **agents trained only to survive** (without any rules or rewards for flocking) will **still show flocking behavior**.  
They build a system called **SELFish**, and show that **agents form flocks** even when the only goal is to **stay alive as long as possible**.

**Limitations:**
1. **Only one agent learned:**  
   - During training, **only one agent was learning**, while the others were just copies.  
   - There was **no real multi-agent learning** or interaction between different agents.  
   - This means they could not test how different strategies might work together (or against each other).

2. **Too simple environment:**  
   - No obstacles, no hills, no places to hide — just an open, flat space.  
   - Agents only had one goal: **stay alive**.  
   - There were no other challenges like **finding food**, **moving to new areas**, or **sharing resources**, which are common in real animal groups.



**Collective Adaptation in Multi-Agent Systems: How Predator Confusion Shapes Swarm-Like Behaviors**  
[https://arxiv.org/abs/2209.06338](https://arxiv.org/abs/2209.06338)

This paper looks at how **predator confusion** can make group behaviors appear.  
They ran tests with **15 or 60 agents** and **one predator**.  
They compared two ways of seeing:
- **GOM (Global Observation Model):** Agents can always see the predator.
- **LOM (Local Observation Model):** Agents can only see nearby things.

**Limitations:**
- **All agents shared one brain:**  
  - Every agent used the **same trained policy**.  
  - They did not learn separately.  
  - As a result, group behaviors were really just **copies of one agent’s behavior**, not real teamwork.

- **Fake group behavior:**  
  - Because all agents used the same brain, their group actions looked organized, but were not the result of real cooperation or decision-making between different agents.



**In summary:**  
Both papers confuse **copying the same behavior** with **real emergent teamwork**.  
Because they use **only one shared policy** and **too simple environments**, they miss the chance to show **true complex group behaviors** like those seen in nature.




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

