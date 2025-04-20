import copy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class FixedOpponentWrapper:
    def __init__(self, env, agent_id, opponent_policy=None):
        self.env = env
        self.agent_id = agent_id
        self.observation_space = env.observation_spaces[agent_id]
        self.action_space = env.action_spaces[agent_id]
        self.opponent_policy = opponent_policy
        self.opponent_id = [a for a in env.agents if a != agent_id][0]
        
    def reset(self):
        self.obs, _ = self.env.reset()
        return self.obs[self.agent_id]
        
    def step(self, action):
        actions = {}
        actions[self.agent_id] = action
        
        # 如果有对手策略，使用它来选择动作
        if self.opponent_policy:
            opponent_action = self.opponent_policy(self.obs[self.opponent_id])
            actions[self.opponent_id] = opponent_action
        else:
            # 否则使用随机动作
            actions[self.opponent_id] = self.env.action_spaces[self.opponent_id].sample()
        
        self.obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        return (self.obs[self.agent_id], 
                rewards[self.agent_id], 
                terminations[self.agent_id] | truncations[self.agent_id], 
                {})

# 使用SB3模型的动作选择函数
def get_action_fn(model):
    def action_fn(obs):
        action, _ = model.predict(obs, deterministic=False)
        return action
    return action_fn

# 开始训练循环
prey1_policy = None
prey2_policy = None

# 训练多个迭代
for iteration in range(5):
    # 训练prey_1，同时使用prey_2的当前策略
    def make_env1():
        multi_env = env(world_name="21_05", use_lppos=False, use_predator=True)
        return FixedOpponentWrapper(multi_env, "prey_1", 
                                   opponent_policy=get_action_fn(prey2_policy) if prey2_policy else None)
    
    env1 = DummyVecEnv([make_env1])
    model1 = PPO("MlpPolicy", env1, verbose=1)
    model1.learn(total_timesteps=50000)
    # 更新prey_1的策略
    prey1_policy = copy.deepcopy(model1)
    
    # 训练prey_2，同时使用prey_1的当前策略
    def make_env2():
        multi_env = env(world_name="21_05", use_lppos=False, use_predator=True)
        return FixedOpponentWrapper(multi_env, "prey_2", 
                                  opponent_policy=get_action_fn(prey1_policy) if prey1_policy else None)
    
    env2 = DummyVecEnv([make_env2])
    model2 = PPO("MlpPolicy", env2, verbose=1)
    model2.learn(total_timesteps=50000)
    # 更新prey_2的策略
    prey2_policy = copy.deepcopy(model2)
    
    # 保存当前迭代的模型
    model1.save(f"prey1_policy_iteration_{iteration}")
    model2.save(f"prey2_policy_iteration_{iteration}")