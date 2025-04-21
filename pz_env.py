from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import numpy as np
import random
from env import DualEvadeObservation, DualEvadeEnv

class MultiDualEvadeEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for DualEvadeEnv.
    Wraps existing DualEvadeEnv to expose both prey agents as independent agents.
    """
    metadata = {"render_modes": ["human"], "name": "multi_dual_evade_parallel"}

    def __init__(self, **dual_evade_kwargs):
        super().__init__()
        # instantiate the original single-agent env
        from env import DualEvadeEnv
        self.inner_env = DualEvadeEnv(**dual_evade_kwargs)
        
        # Create another observation object for the second agent
        self.prey2_observation = DualEvadeObservation()
        
        # Store the reward function
        self.reward_function = self.inner_env.reward_function

        # define agent names
        self.agents = ["prey_1", "prey_2"]
        self.possible_agents = list(self.agents)

        # use same observation and action space for both
        obs_space = self.inner_env.observation_space
        act_space = self.inner_env.action_space
        self.observation_spaces = {agent: obs_space for agent in self.agents}
        self.action_spaces = {agent: act_space for agent in self.agents}
        
        # Store the action type
        self.action_type = self.inner_env.action_type

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        # Reset inner env
        obs0, _ = self.inner_env.reset(options=options, seed=seed)
        # Get initial observations for both agents
        self.agents = list(self.possible_agents)
        
        # 确保所有观测都是正确的numpy数组
        prey1_obs = np.array(obs0, dtype=np.float32)
        
        # 获取prey_2的观测
        prey2_obs_obj = self.__get_prey2_observation()
        prey2_obs = np.array(prey2_obs_obj, dtype=np.float32)
        
        # 创建观测字典
        observations = {
            "prey_1": prey1_obs,
            "prey_2": prey2_obs
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # 处理智能体的动作
        action_prey1 = actions["prey_1"]
        action_prey2 = actions["prey_2"]
        
        # 同时为两个智能体设置动作，通过inner_env的内部模型
        # 直接访问inner_env的模型属性进行操作
        model = self.inner_env.model
        
        # 根据动作类型设置目标位置
        if self.action_type == DualEvadeEnv.ActionType.DISCRETE:
            action_list = self.inner_env.action_list
            # 同时设置两个智能体的目标位置（离散动作）
            model.prey_1.set_destination(action_list[action_prey1])
            model.prey_2.set_destination(action_list[action_prey2])
        else:
            # 连续动作直接使用动作向量
            model.prey_1.set_destination(tuple(action_prey1))
            model.prey_2.set_destination(tuple(action_prey2))
        
        # 直接推进环境模拟一步
        model_t = self.inner_env.model.time + self.inner_env.time_step
        while model.running and model.time < model_t:
            model.step()
        
        # 更新步数计数
        self.inner_env.step_count += 1
        truncated = (self.inner_env.step_count >= self.inner_env.max_step)
        
        # 检查目标完成情况
        if model.prey_data_1.goal_achieved and model.prey_data_2.goal_achieved:
            model.stop()
        
        # 获取两个智能体的观测
        prey1_obs_obj = self.inner_env.__update_observation__(
            observation=self.inner_env.observation,
            prey=model.prey_1,
            prey_data=model.prey_data_1,
            other=model.prey_2
        )
        
        prey2_obs_obj = self.__get_prey2_observation()
        
        # 计算奖励
        reward_prey1 = self.reward_function(prey1_obs_obj)
        reward_prey2 = self.reward_function(prey2_obs_obj)
        
        # 转换为numpy数组
        prey1_obs = np.array(prey1_obs_obj, dtype=np.float32)
        prey2_obs = np.array(prey2_obs_obj, dtype=np.float32)
        
        # 构建并行环境返回值
        observations = {
            "prey_1": prey1_obs,
            "prey_2": prey2_obs
        }
        rewards = {
            "prey_1": reward_prey1,
            "prey_2": reward_prey2
        }
        
        # 确定环境是否终止
        terminated = not model.running
        
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        
        # 构建信息字典
        if terminated or truncated:
            info = {
                "prey_1": {
                    "captures": model.prey_data_1.puff_count,
                    "is_success": model.prey_data_1.goal_achieved and model.prey_data_1.puff_count == 0,
                    "survived": model.prey_data_1.puff_count == 0
                },
                "prey_2": {
                    "captures": model.prey_data_2.puff_count,
                    "is_success": model.prey_data_2.goal_achieved and model.prey_data_2.puff_count == 0,
                    "survived": model.prey_data_2.puff_count == 0
                }
            }
        else:
            info = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, info
        
    def __get_prey2_observation(self):
        """Generate the observation from prey_2's perspective"""
        # Get the model from inner_env
        model = self.inner_env.model
        
        # Update observation for prey_2 perspective
        self.inner_env.__update_observation__(
            observation=self.prey2_observation,
            prey=model.prey_2,
            prey_data=model.prey_data_2,
            other=model.prey_1
        )
        
        return self.prey2_observation.copy()

    def render(self):
        # delegate to inner env's render via model
        return None

    def close(self):
        self.inner_env.close()

# wrapper function for default usage
# No OrderEnforcingWrapper needed for ParallelEnv

def env(**kwargs):
    return MultiDualEvadeEnv(**kwargs)

def random_policy(env, num_episodes=5, max_steps=300, render=True):
    """
    实现多智能体环境的随机策略
    
    参数:
        env: 多智能体环境(PettingZoo ParallelEnv)
        num_episodes: 要运行的回合数
        max_steps: 每个回合的最大步数
        render: 是否渲染环境
    
    返回:
        每个智能体每个回合的总奖励和成功率
    """
    # 统计数据
    episode_rewards = {agent: [] for agent in env.possible_agents}
    episode_success = {agent: 0 for agent in env.possible_agents}
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        # 重置环境
        observations, infos = env.reset()
        
        # 初始化回合统计
        step_count = 0
        episode_reward = {agent: 0 for agent in env.agents}
        done = False
        
        while not done and step_count < max_steps:
            # 为每个智能体采样随机动作
            actions = {}
            
            for agent in env.agents:
                action_space = env.action_spaces[agent]
                
                # 根据动作空间类型采样动作
                if isinstance(action_space, spaces.Discrete):
                    # 离散动作空间
                    actions[agent] = action_space.sample()
                else:
                    # 连续动作空间
                    actions[agent] = action_space.sample()
            
            # 执行动作
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 更新统计
            for agent in env.agents:
                episode_reward[agent] += rewards[agent]
            
            # 检查是否完成
            done = all(terminations.values()) or all(truncations.values())
            step_count += 1
            
            # 渲染环境
            if render:
                env.render()
        
        # 记录回合结果
        for agent in env.agents:
            episode_rewards[agent].append(episode_reward[agent])
            if agent in infos and "is_success" in infos[agent] and infos[agent]["is_success"]:
                episode_success[agent] += 1
        
        # 打印回合结果
        print(f"Step count: {step_count}")
        for agent in env.agents:
            print(f"{agent} - Reward: {episode_reward[agent]:.4f}")
    
    # 计算统计数据
    for agent in env.possible_agents:
        avg_reward = np.mean(episode_rewards[agent]) if episode_rewards[agent] else 0
        success_rate = episode_success[agent] / num_episodes if num_episodes > 0 else 0
        print(f"\n{agent} - 统计:")
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  成功率: {success_rate:.2f}")
    
    return episode_rewards, episode_success

