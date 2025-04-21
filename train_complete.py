import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import os
from pz_env import env as make_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Batch
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from env import DualEvadeEnv


def custom_reward(obs):
    """
    Simple reward function for debugging:
    - Penalty for getting puffed
    - Big reward for reaching goal
    """
    if obs.puffed > 0:
        return -1.0
    
    if obs.prey_goal_distance < 0.1:
        return 1.0
    return 0.0 

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# 定义直接的PettingZoo Parallel环境封装
class ParallelPettingZooEnv:
    """封装PettingZoo的Parallel环境"""
    
    def __init__(self, env):
        self.env = env
        # 设置当前智能体、观测和动作空间
        self.agents = self.env.agents
        self.agent_ids = {agent: i for i, agent in enumerate(self.agents)}
        
        # 缓存当前智能体的观测和信息
        self.observations = None
        self.infos = None
        
        # 获取观测空间和动作空间
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
    
    def reset(self, seed=None):
        """重置环境并返回所有智能体的观测"""
        self.observations, self.infos = self.env.reset(seed=seed)
        return self.observations, self.infos
    
    def step(self, actions):
        """并行执行所有智能体的动作"""
        # 转换动作为字典格式，如果不是
        if not isinstance(actions, dict):
            action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        else:
            action_dict = actions
            
        # 执行环境步
        self.observations, rewards, terminations, truncations, self.infos = self.env.step(action_dict)
        
        # 检查是否所有智能体都终止
        done = all(terminations.values()) or all(truncations.values())
        
        return self.observations, rewards, terminations, truncations, self.infos, done
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()

# 创建环境
def make_env_fn(world="21_05", use_pred=True, render=False):
    def _make_env():
        # 创建基础环境
        raw_env = make_env(
            world_name=world,
            use_lppos=True,
            use_predator=use_pred,  # 默认值已改为True
            max_step=300,
            reward_function=custom_reward,
            time_step=0.25,
            render=render,
            real_time=render,
            end_on_pov_goal=False,  # 要求两个智能体都达到目标才结束
            use_other=True,
            action_type=DualEvadeEnv.ActionType.CONTINUOUS
        )
        # 直接使用Parallel环境，不转换为AEC
        return ParallelPettingZooEnv(raw_env)
    return _make_env

# 创建向量化环境
def make_vec_env(training=True, render=False):
    return DummyVectorEnv([make_env_fn(render=render and not training)])

# 创建网络
def make_net(state_dim, action_dim, device, actor_hidden=[256, 256], critic_hidden=[256, 256]):
    # 特征网络基础
    net_a = Net(
        state_shape=state_dim,
        hidden_sizes=actor_hidden,
        activation=nn.ReLU,
        device=device
    )
    net_c = Net(
        state_shape=state_dim, 
        hidden_sizes=critic_hidden,
        activation=nn.ReLU,
        device=device
    )
    
    # 构建Actor和Critic
    actor = Actor(
        net_a,
        action_dim,
        device=device,
        softmax_output=True
    )
    critic = Critic(
        net_c,
        device=device
    )
    
    return actor, critic

# 创建一个PPOPolicy的包装器，添加set_agent_id方法和熵系数动态调整
class AgentPPOPolicy(PPOPolicy):
    def __init__(self, *args, **kwargs):
        # 确保action_scaling=False用于离散动作空间
        if 'action_scaling' not in kwargs:
            kwargs['action_scaling'] = False
        
        super().__init__(*args, **kwargs)
        self.agent_id = None
    
    def set_agent_id(self, agent_id):
        self.agent_id = agent_id
        print(f"设置智能体ID: {agent_id}")

# 创建PPO策略
def make_ppo_policy(actor, critic, action_space, device='cpu', lr=3e-4):
    # 优化器
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr
    )
    
    # 创建PPO策略
    dist_fn = torch.distributions.Categorical  # 用于离散动作空间
    policy = AgentPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=action_space,
        discount_factor=0.99,
        gae_lambda=0.95,
        reward_normalization=True,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.00,        # 使用固定熵系数
        eps_clip=0.2,
        value_clip=True,
        action_scaling=False,  # 对离散动作空间必须设为False
        deterministic_eval=True,
        advantage_normalization=True
    )
    
    return policy

def train_parallel(args):
    """使用并行动作的训练方法"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    
    # 创建环境 - 使用传入的环境创建函数
    if hasattr(args, 'custom_env_fn'):
        env = args.custom_env_fn(render=False)
        print("使用自定义环境创建函数")
    else:
        # 默认使用标准环境创建函数
        use_predator = not getattr(args, 'no_predator', False)
        env = make_env_fn(world="21_05", use_pred=use_predator, render=False)()
        print(f"使用标准环境创建函数，捕食者：{'启用' if use_predator else '禁用'}")
    
    print(f"智能体列表: {env.agents}")
    
    # 获取观测和动作空间
    obs, _ = env.reset()
    
    # 获取观测维度和动作维度
    example_obs = obs[env.agents[0]]
    obs_shape = len(example_obs)
    action_dim = env.action_spaces[env.agents[0]].n
    
    print(f"观测维度: {obs_shape}, 动作空间: {action_dim}")
    
    # 创建智能体策略
    agents = {}
    for agent_id in env.agents:
        print(f"为智能体 {agent_id} 创建策略...")
        # 创建网络
        actor, critic = make_net(
            state_dim=obs_shape, 
            action_dim=action_dim,
            device=device
        )
        
        # 创建PPO策略
        policy = make_ppo_policy(
            actor=actor,
            critic=critic,
            action_space=env.action_spaces[agent_id],
            device=device,
            lr=args.lr
        )
        
        # 设置智能体ID
        policy.set_agent_id(agent_id)
        
        agents[agent_id] = policy
    
    # 创建日志记录器
    os.makedirs("./log/parallel_ppo", exist_ok=True)
    writer = SummaryWriter(f"./log/parallel_ppo/{int(time.time())}")
    
    # 记录训练数据
    observations = {agent_id: [] for agent_id in env.agents}
    actions = {agent_id: [] for agent_id in env.agents}
    rewards = {agent_id: [] for agent_id in env.agents}
    next_observations = {agent_id: [] for agent_id in env.agents}
    dones = {agent_id: [] for agent_id in env.agents}
    log_probs = {agent_id: [] for agent_id in env.agents}
    
    # 训练循环
    total_steps = 0
    total_episodes = 0
    
    # 为每个智能体单独记录最佳奖励
    best_rewards = {agent_id: float('-inf') for agent_id in env.agents}
    
    # 记录每个epoch的平均奖励
    epoch_rewards = {agent_id: [] for agent_id in env.agents}
    
    # 确保模型保存目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print("开始训练...")
    for epoch in range(args.epoch):
        # 打印当前epoch
        print(f"\n开始 Epoch {epoch+1}/{args.epoch}")
        
        # 重置当前epoch的奖励统计
        episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
        episode_steps = 0
        epoch_total_rewards = {agent_id: 0.0 for agent_id in env.agents}
        epoch_episodes = 0
        
        # 重置环境获得初始观测
        obs, _ = env.reset()
        
        # 每个epoch收集若干steps的数据
        for step in range(args.step_per_epoch):
            # 为每个智能体选择动作
            actions_dict = {}
            log_prob_dict = {}
            
            for agent_id in env.agents:
                agent = agents[agent_id]
                # 提取观测张量 - 观测已经是numpy数组
                obs_tensor = torch.tensor(obs[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                
                # 创建batch
                batch = Batch(
                    obs=obs_tensor,
                    info={}
                )
                
                # 选择动作
                with torch.no_grad():
                    logits, _ = agent.actor(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    act = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor([act], device=device)).item()
                
                # 存储当前状态的数据
                observations[agent_id].append(obs[agent_id].copy())
                actions[agent_id].append(act)
                log_probs[agent_id].append(log_prob)
                
                # 将动作添加到动作字典
                actions_dict[agent_id] = act
                log_prob_dict[agent_id] = log_prob
            
            # 执行所有智能体的动作
            next_obs, rewards_dict, terminations, truncations, infos, done = env.step(actions_dict)
            
            # 更新统计信息和存储数据
            for agent_id in env.agents:
                rewards[agent_id].append(rewards_dict[agent_id])
                next_observations[agent_id].append(next_obs[agent_id].copy())
                dones[agent_id].append(terminations[agent_id] or truncations[agent_id])
                episode_rewards[agent_id] += rewards_dict[agent_id]
            
            episode_steps += 1
            total_steps += 1
            
            # 移至下一状态
            obs = next_obs
            
            # 如果回合结束，重置环境
            if done:
                obs, _ = env.reset()
                
                # 记录回合奖励
                total_episodes += 1
                epoch_episodes += 1
                
                # 累加每个智能体的奖励到当前epoch总奖励
                for agent_id in env.agents:
                    epoch_total_rewards[agent_id] += episode_rewards[agent_id]
                    
                    # 记录到TensorBoard
                    writer.add_scalar(f"train/{agent_id}_episode_reward", episode_rewards[agent_id], total_episodes)
                
                # 计算并记录平均奖励
                avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
                writer.add_scalar("train/avg_episode_reward", avg_reward, total_episodes)
                writer.add_scalar("train/episode_length", episode_steps, total_episodes)
                
                # 为每个智能体单独检查和保存最佳模型
                for agent_id in env.agents:
                    if episode_rewards[agent_id] > best_rewards[agent_id]:
                        best_rewards[agent_id] = episode_rewards[agent_id]
                        torch.save(agents[agent_id].state_dict(), f"{args.save_path}{agent_id}_best.pth")
                        print(f"保存 {agent_id} 最佳模型，奖励: {best_rewards[agent_id]:.2f}")
                
                # 打印当前回合信息
                if total_episodes % 10 == 0:  # 每10个回合打印一次详细信息
                    print(f"Episode {total_episodes}, 步数: {episode_steps}")
                    for agent_id in env.agents:
                        print(f"  {agent_id} 奖励: {episode_rewards[agent_id]:.2f}")
                
                # 重置回合统计信息
                episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
                episode_steps = 0
        
        # 计算当前epoch的平均奖励
        if epoch_episodes > 0:
            for agent_id in env.agents:
                avg_epoch_reward = epoch_total_rewards[agent_id] / epoch_episodes
                epoch_rewards[agent_id].append(avg_epoch_reward)
                
                # 记录到TensorBoard
                writer.add_scalar(f"train/{agent_id}_epoch_avg_reward", avg_epoch_reward, epoch)
            
            # 打印当前epoch的平均奖励
            print(f"\n===== Epoch {epoch+1}/{args.epoch} 统计信息 =====")
            print(f"完成回合数: {epoch_episodes}, 总步数: {total_steps}")
            print("各智能体平均奖励:")
            for agent_id in env.agents:
                print(f"  {agent_id}: {epoch_rewards[agent_id][-1]:.2f}")
            print("===================================\n")
        
        # 每个epoch结束，对每个智能体更新策略
        for agent_id, agent in agents.items():
            if len(observations[agent_id]) > 0:
                # 收集足够的数据更新一次
                print(f"更新智能体 {agent_id} 策略，数据量: {len(observations[agent_id])}")
                
                # 转换为张量
                obs_tensor = torch.tensor(
                    np.array(observations[agent_id]), 
                    dtype=torch.float32, 
                    device=device
                )
                act_tensor = torch.tensor(
                    np.array(actions[agent_id]), 
                    dtype=torch.long, 
                    device=device
                )
                rew_tensor = torch.tensor(
                    np.array(rewards[agent_id]), 
                    dtype=torch.float32, 
                    device=device
                )
                next_obs_tensor = torch.tensor(
                    np.array(next_observations[agent_id]), 
                    dtype=torch.float32, 
                    device=device
                )
                done_tensor = torch.tensor(
                    np.array(dones[agent_id]), 
                    dtype=torch.bool, 
                    device=device
                )
                
                # 计算回报
                returns = []
                advantages = []
                
                # 计算状态值预测
                with torch.no_grad():
                    values = agent.critic(obs_tensor).flatten().cpu().numpy()
                    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
                    next_values = agent.critic(next_obs_tensor).flatten().cpu().numpy()
                
                # 计算GAE (Generalized Advantage Estimation) 和 returns
                last_gae = 0.0
                for i in reversed(range(len(rewards[agent_id]))):
                    r = rewards[agent_id][i]
                    d = dones[agent_id][i]
                    
                    # 如果是最后一步，下一步值为0
                    if i == len(rewards[agent_id]) - 1:
                        next_val = 0.0
                    else:
                        next_val = next_values[i + 1]
                    
                    delta = r + args.gamma * next_val * (1 - d) - values[i]
                    last_gae = delta + args.gamma * 0.95 * (1 - d) * last_gae
                    advantages.insert(0, last_gae)
                    
                    # 计算回报
                    if i == len(rewards[agent_id]) - 1:
                        running_return = r
                    else:
                        running_return = r + args.gamma * running_return * (1 - d)
                    returns.insert(0, running_return)
                
                returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
                advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
                
                # 标准化回报
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
                
                # 标准化优势
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                
                # 转换log_probs为张量
                log_probs_tensor = torch.tensor(
                    np.array(log_probs[agent_id]), 
                    dtype=torch.float32, 
                    device=device
                )
                
                # 创建batch
                batch = Batch(
                    obs=obs_tensor,
                    act=act_tensor,
                    rew=rew_tensor,
                    done=done_tensor,
                    obs_next=next_obs_tensor,
                    returns=returns_tensor,
                    adv=advantages_tensor,
                    logp_old=log_probs_tensor,
                    v_s=values_tensor,
                    info={}
                )
                
                # PPO需要batch_size和repeat参数
                batch_size = min(64, len(observations[agent_id])) 
                agent.learn(batch, batch_size=batch_size, repeat=4)  
                
                # 清空数据
                observations[agent_id] = []
                actions[agent_id] = []
                rewards[agent_id] = []
                next_observations[agent_id] = []
                dones[agent_id] = []
                log_probs[agent_id] = []
    
    # 训练结束，保存最终模型和性能曲线
    for agent_id, agent in agents.items():
        # 保存最终模型
        torch.save(agent.state_dict(), f"{args.save_path}{agent_id}_final.pth")
        
        # 如果有足够的epoch，绘制学习曲线
        if len(epoch_rewards[agent_id]) > 1:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(epoch_rewards[agent_id])+1), epoch_rewards[agent_id])
            plt.xlabel('Epoch')
            plt.ylabel('Average Reward')
            plt.title(f'{agent_id} Learning Curve')
            plt.savefig(f"{args.save_path}{agent_id}_learning_curve.png")
            plt.close()
    
    # 打印每个智能体的最佳奖励
    print("\n训练完成! 各智能体最佳奖励:")
    for agent_id, best_reward in best_rewards.items():
        print(f"  {agent_id}: {best_reward:.2f}")
    
    # 关闭环境
    env.close()
    writer.close()
    
    return agents

def test_parallel(args):
    """测试训练好的智能体，使用并行动作"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境 - 使用传入的环境创建函数或默认环境
    if hasattr(args, 'custom_env_fn'):
        env = args.custom_env_fn(render=args.render_test)
        print("使用自定义环境创建函数")
    else:
        # 默认使用标准环境创建函数
        use_predator = not getattr(args, 'no_predator', False)
        env = make_env_fn(world="21_05", use_pred=use_predator, render=args.render_test)()
        print(f"使用标准环境创建函数，捕食者：{'启用' if use_predator else '禁用'}")
    
    print(f"智能体列表: {env.agents}")
    
    # 获取观测和动作空间
    obs, _ = env.reset()
    
    # 获取观测维度和动作维度
    example_obs = obs[env.agents[0]]
    obs_shape = len(example_obs)
    action_dim = env.action_spaces[env.agents[0]].n
    
    # 创建智能体策略
    agents = {}
    models_loaded = True  # 跟踪是否所有模型都成功加载
    
    for agent_id in env.agents:
        # 创建网络
        actor, critic = make_net(
            state_dim=obs_shape, 
            action_dim=action_dim,
            device=device
        )
        
        # 创建PPO策略
        policy = make_ppo_policy(
            actor=actor,
            critic=critic,
            action_space=env.action_spaces[agent_id],
            device=device,
            lr=args.lr
        )
        
        # 尝试加载最终模型，如果不存在，尝试加载最佳模型
        models_found = False
        for model_type in ["final", "best"]:
            model_path = f"{args.save_path}{agent_id}_{model_type}.pth"
            try:
                policy.load_state_dict(torch.load(model_path, map_location=device))
                print(f"成功加载 {agent_id} 的{model_type}模型: {model_path}")
                models_found = True
                break
            except FileNotFoundError:
                print(f"未找到 {model_path}")
        
        if not models_found:
            print(f"警告: {agent_id} 没有找到训练好的模型，将使用随机策略！")
            models_loaded = False
        
        agents[agent_id] = policy
    
    if not models_loaded:
        print("\n警告: 至少有一个智能体使用随机策略。测试结果可能无法反映实际训练效果。")
        response = input("是否继续测试? (y/n): ")
        if response.lower() not in ["y", "yes"]:
            print("测试取消")
            return
    
    # 测试多个回合
    total_rewards = {agent_id: 0.0 for agent_id in env.agents}
    goal_achieved = {agent_id: 0 for agent_id in env.agents}
    captured = {agent_id: 0 for agent_id in env.agents}
    total_steps = 0
    episodes = args.test_num
    
    for episode in range(episodes):
        episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
        episode_steps = 0
        
        # 重置环境
        obs, _ = env.reset()
        done = False
        
        while not done:
            # 为每个智能体选择动作
            actions_dict = {}
            
            # 获取每个智能体的动作
            for agent_id in env.agents:
                agent = agents[agent_id]
                agent_obs = obs[agent_id]
                
                # 打印智能体位置
                if args.verbose and episode_steps % 10 == 0:
                    print(f"{agent_id} 位置: ({agent_obs[0]:.3f}, {agent_obs[1]:.3f}), 目标距离: {agent_obs[9]:.3f}")
                
                # 选择动作
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    logits, _ = agent.actor(obs_tensor)
                    if args.deterministic:
                        act = torch.argmax(logits, dim=1).item()
                    else:
                        act = torch.distributions.Categorical(logits=logits).sample().item()
                
                actions_dict[agent_id] = act
            
            # 执行所有智能体的动作
            next_obs, rewards, terminations, truncations, infos, done = env.step(actions_dict)
            
            # 更新统计信息
            for agent_id in env.agents:
                episode_rewards[agent_id] += rewards[agent_id]
            
            episode_steps += 1
            total_steps += 1
            
            # 如果测试时需要渲染
            if args.render_test:
                env.render()
            
            # 更新观测
            obs = next_obs
        
        # 回合结束，收集额外信息
        for agent_id in env.agents:
            # 检查智能体是否到达目标或被捕获
            if agent_id in infos and infos[agent_id]:
                if "is_success" in infos[agent_id] and infos[agent_id]["is_success"]:
                    goal_achieved[agent_id] += 1
                if "captures" in infos[agent_id] and infos[agent_id]["captures"] > 0:
                    captured[agent_id] += 1
        
        # 输出回合统计信息
        print(f"回合 {episode + 1}/{episodes} - 步数: {episode_steps}")
        print("各智能体当前回合奖励:")
        for a_id, a_reward in episode_rewards.items():
            print(f"  {a_id}: {a_reward:.2f}")
            total_rewards[a_id] += a_reward
        avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
        print(f"  平均: {avg_reward:.2f}")
    
    # 输出平均统计
    print("\n测试结果总结:")
    print(f"总回合数: {episodes}, 总步数: {total_steps}")
    print("各智能体平均奖励:")
    for agent_id in env.agents:
        avg_reward = total_rewards[agent_id] / episodes
        print(f"  {agent_id}: {avg_reward:.2f}")
    
    print("\n目标达成次数:")
    for agent_id in env.agents:
        success_rate = goal_achieved[agent_id] / episodes * 100
        print(f"  {agent_id}: {goal_achieved[agent_id]}/{episodes} ({success_rate:.1f}%)")
    
    print("\n被捕获次数:")
    for agent_id in env.agents:
        capture_rate = captured[agent_id] / episodes * 100
        print(f"  {agent_id}: {captured[agent_id]}/{episodes} ({capture_rate:.1f}%)")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", type=str, default="./models/")
    parser.add_argument("--test-num", type=int, default=3)
    parser.add_argument("--render-test", action="store_true", default=False)
    parser.add_argument("--test-only", action="store_true", default=False)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--no-predator", action="store_true", default=False, help="禁用捕食者")
    args = parser.parse_args()
    
    # 设置环境参数
    use_predator = not args.no_predator
    
    # 创建定制环境函数
    def custom_env_fn(render=False):
        return make_env_fn(world="21_05", use_pred=use_predator, render=render)()
    
    # 将环境函数附加到args
    args.custom_env_fn = custom_env_fn
    
    # 如果是测试模式
    if args.test_only:
        test_parallel(args)
    else:
        # 训练并测试
        agents = train_parallel(args)
        # 测试
        args.render_test = True
        test_parallel(args) 