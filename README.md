
## Related Prior Work

**Emergent Escape-based Flocking Behavior using Multi-Agent Reinforcement Learning**  
[https://arxiv.org/abs/1905.04077](https://arxiv.org/abs/1905.04077)

This paper asks if **agents trained only to survive** (without any rules or rewards for flocking) will **still show flocking behavior**.  
They build a system called **SELFish**, and show that **agents form flocks** even when the only goal is to **stay alive as long as possible**.

**Limitations:**

 **Only one agent learned:** During training, **only one agent was learning**, while the others were just copies.  There was **no real multi-agent learning** or interaction between different agents. This means they could not test how different strategies might work together (or against each other).

 **Too simple environment:**
   - No obstacles, no hills, no places to hide — just an open, flat space.  Agents only had one goal: **stay alive**. There were no other challenges like **finding food**, **moving to new areas**.



**Collective Adaptation in Multi-Agent Systems: How Predator Confusion Shapes Swarm-Like Behaviors**  

[https://arxiv.org/abs/2209.06338](https://arxiv.org/abs/2209.06338)

This paper looks at how **predator confusion** can make group behaviors appear.  
They ran tests with **15 or 60 agents** and **one predator**.  
They compared two ways of seeing:
- **GOM (Global Observation Model):** Agents can always see the predator.
- **LOM (Local Observation Model):** Agents can only see nearby things.

**Limitations:**
- **All agents shared one brain:**  
  - Every agent used the **same trained policy**. They did not learn separately. As a result, group behaviors were really just **copies of one agent’s behavior**, not real teamwork.
-  Because all agents used the same brain, their group actions looked organized, but were not the result of real cooperation or decision-making between different agents.



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


# SAC and DQN 

When there is no predator: very stable and they will eventually share the simliar policy and behavior:

![image](https://github.com/user-attachments/assets/318b7706-4f31-42bc-be63-9af6ceb5a480)

<img width="500" alt="image" src="https://github.com/user-attachments/assets/b6dabf15-73c1-4cf1-a2ad-c6c232f97f9a" />
<img width="500" alt="image" src="https://github.com/user-attachments/assets/8745c809-fd45-41ee-9871-a5c3e849b55e" />



But when there is predator, their performance and training will be very different. Even under two setting: 

They don't see each other: 

![image](https://github.com/user-attachments/assets/f1fee172-2dc4-470c-a0e2-db422cdfeb09)



They see each other: 

![image](https://github.com/user-attachments/assets/30f19f92-04bd-4b42-95c4-192effe1c94a)

<img width="195" alt="image" src="https://github.com/user-attachments/assets/6e728b65-76e0-4b99-8a63-6ca3d6d54d67" />
<img width="195" alt="image" src="https://github.com/user-attachments/assets/91998b3e-b0dc-4bc8-988b-2bddd7d3b80a" />
<img width="195" alt="image" src="https://github.com/user-attachments/assets/1501fc78-a0a1-4eff-9cde-bc317731b93d" />
<img width="195" alt="image" src="https://github.com/user-attachments/assets/e5ef1615-df0e-4013-bd64-857cc118d243" />
<img width="195" alt="image" src="https://github.com/user-attachments/assets/ed2b4c85-62f6-4e3a-ad02-258783f6adf4" />

CHecklist: Paraller env setting, larger batchsize, smaller learning rate.


[https://openreview.net/forum?id=SJxu5iR9KQ&utm_source=chatgpt.com](https://openreview.net/forum?id=SJxu5iR9KQ)
https://arxiv.org/abs/2401.07056
