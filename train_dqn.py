import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import os
from pz_env import env as make_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Batch, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net
from env import DualEvadeEnv


import math

def custom_reward(obs):
    """
    obs.self_x, obs.self_y           -- agent 的当前坐标
    obs.predator_x, obs.predator_y   -- 捕食者坐标（为 0 时表示不可见）
    """
    reward = 0.0
    # 自身坐标
    x, y = obs.self_x, obs.self_y

    # 捕食惩罚：只有 predator 坐标非 0 时才判断距离
    if obs.predator_x != 0 or obs.predator_y != 0:
        d_pred = math.hypot(x - obs.predator_x, y - obs.predator_y)
        if d_pred < 0.1:
            reward -= 1.0

    # 成功奖励：到达 (1.0, 0.5) 半径 0.1 内
    d_goal = math.hypot(x - 1.0, y - 0.5)
    if d_goal < 0.1:
        reward += 1.0

    return reward

# 自定义多智能体策略管理器
class MultiAgentDQNPolicyManager:
    """
    多智能体DQN策略管理器，用于处理多个智能体的策略
    """
    def __init__(self, policies):
        """
        初始化多智能体策略管理器
        policies: 字典，键为智能体ID，值为对应的策略
        """
        self.policies = policies
        self.agent_ids = list(policies.keys())
    
    def forward(self, batch, state=None, **kwargs):
        """
        根据智能体ID分发观测到对应的策略
        """
        results = {}
        
        # 对每个智能体进行前向传播
        for agent_id in self.agent_ids:
            if agent_id in batch:
                policy = self.policies[agent_id]
                results[agent_id] = policy(batch[agent_id], state, **kwargs)
        
        return Batch(results)
    
    def learn(self, batch, **kwargs):
        """
        训练所有策略
        """
        results = {}
        total_loss = 0.0
        
        # 为每个智能体独立更新策略
        for agent_id, policy in self.policies.items():
            if agent_id in batch:
                result = policy.learn(batch[agent_id], **kwargs)
                results[agent_id] = result
                if isinstance(result, dict) and "loss" in result:
                    total_loss += result["loss"]
                else:
                    total_loss += result
        
        # 返回所有策略的平均损失
        return total_loss / len(self.policies) if self.policies else 0.0
    
    def set_eps(self, eps):
        """设置所有策略的探索率"""
        for policy in self.policies.values():
            policy.set_eps(eps)
    
    def train(self):
        """设置为训练模式"""
        for policy in self.policies.values():
            policy.train()
    
    def eval(self):
        """设置为评估模式"""
        for policy in self.policies.values():
            policy.eval()
    
    def update(self, sample_size=10, **kwargs):
        """更新所有策略"""
        losses = []
        for policy in self.policies.values():
            loss = policy.update(sample_size=sample_size, **kwargs)
            losses.append(loss)
        return sum(losses) / len(losses) if losses else 0.0

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        print("GPU is not available, using CPU instead")

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
            use_predator=use_pred,
            max_step=50,
            reward_function=custom_reward,
            time_step=0.25,
            render=render,
            real_time=render,
            end_on_pov_goal=False,  # 要求两个智能体都达到目标才结束
            use_other=True,
            action_type=DualEvadeEnv.ActionType.DISCRETE  # 对DQN使用离散动作空间
        )
        # 直接使用Parallel环境，不转换为AEC
        return ParallelPettingZooEnv(raw_env)
    return _make_env

# 为DQN创建离散网络
def make_dqn_net(state_dim, action_dim, device, hidden_sizes=[256, 256]):
    net = Net(
        state_shape=state_dim,
        action_shape=action_dim,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        device=device
    )
    return net

# 创建DQN策略
def make_dqn_policy(net, action_space, device='cpu', lr=3e-4, gamma=0.99, 
                    estimation_step=3, target_update_freq=100):
    # 优化器
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    
    # 创建DQN策略
    policy = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=gamma,
        estimation_step=estimation_step,
        target_update_freq=target_update_freq,
        action_space=action_space,
        # 额外参数
        reward_normalization=False,
        is_double=True  # 使用Double DQN
    )
    
    return policy

def train_dqn_agents(args):
    """并行训练多个DQN智能体"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    use_predator = not getattr(args, 'no_predator', False)
    env = make_env_fn(world="21_05", use_pred=use_predator, render=False)()
    print(f"创建环境成功，智能体列表: {env.agents}")
    
    # 获取观测和动作空间
    obs_dict, _ = env.reset(seed=args.seed)
    
    # 为每个智能体创建策略
    agents = {}
    buffers = {}
    
    for agent_id in env.agents:
        print(f"\n开始为智能体 {agent_id} 设置训练...")
        # 获取观测维度和动作维度
        obs_shape = len(obs_dict[agent_id])
        action_dim = env.action_spaces[agent_id].n
        print(f"观测维度: {obs_shape}, 动作空间: {action_dim}")
        
        # 创建网络
        net = make_dqn_net(
            state_dim=obs_shape, 
            action_dim=action_dim,
            device=device,
            hidden_sizes=args.hidden_sizes
        )
        
        # 创建DQN策略
        policy = DQNPolicy(
            model=net,
            optim=torch.optim.Adam(net.parameters(), lr=args.lr),
            discount_factor=args.gamma,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq,
            action_space=env.action_spaces[agent_id],
            is_double=True  # 使用Double DQN
        )
        
        agents[agent_id] = policy
        
        # 创建每个智能体的回放缓冲区
        buffers[agent_id] = VectorReplayBuffer(
            total_size=args.buffer_size,
            buffer_num=1  # 单个环境实例
        )
    
    # 创建日志记录器
    log_path = os.path.join(args.logdir, 'dqn', f"{int(time.time())}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(args.epoch):
        print(f"\n开始Epoch {epoch+1}/{args.epoch}")
        
        # 设置训练模式
        for agent_id, policy in agents.items():
            policy.train()
            policy.set_eps(args.eps_train)
        
        # 每个智能体的累积回报
        epoch_returns = {agent_id: [] for agent_id in env.agents}
        
        # 先收集预训练数据，填充经验回放缓冲区
        if epoch == 0:
            print("收集初始数据...")
            # 重置环境
            obs_dict, _ = env.reset(seed=args.seed + epoch)
            # 收集一些步骤来填充缓冲区
            for i in range(args.batch_size * 2):
                # 为每个智能体选择动作
                actions = {}
                for agent_id, policy in agents.items():
                    # 转换观测为tensor
                    obs_tensor = torch.tensor(
                        obs_dict[agent_id], 
                        dtype=torch.float32, 
                        device=device
                    ).unsqueeze(0)
                    
                    # 选择动作
                    result = policy(Batch(obs=obs_tensor, info={}))
                    # 确保动作是int类型
                    if torch.is_tensor(result.act):
                        actions[agent_id] = result.act.item()
                    else:
                        actions[agent_id] = int(result.act)
                
                # 执行动作
                next_obs_dict, rewards, terminations, truncations, infos, done = env.step(actions)
                
                # 为每个智能体存储经验
                for agent_id in env.agents:
                    # 计算n步return (简化版的计算)
                    rew = rewards[agent_id]
                    done_flag = terminations[agent_id] or truncations[agent_id]
                    
                    # 这里我们不直接计算returns，让DQN算法内部处理
                    # DQN会根据rew, done和下一个状态计算TD目标
                    # returns字段只是为了兼容性，实际上DQN不会直接使用它
                    
                    # 创建经验
                    experience = Batch(
                        obs=np.array([obs_dict[agent_id]], dtype=np.float32),
                        act=np.array([actions[agent_id]], dtype=np.int64),
                        rew=np.array([rew], dtype=np.float32),
                        terminated=np.array([terminations[agent_id]], dtype=np.bool_),
                        truncated=np.array([truncations[agent_id]], dtype=np.bool_),
                        done=np.array([done_flag], dtype=np.bool_),
                        obs_next=np.array([next_obs_dict[agent_id]], dtype=np.float32),
                        info=np.array([{}]),
                        # 我们还是提供returns字段，但它的值在DQN算法中可能不会直接使用
                        returns=np.array([rew], dtype=np.float32)  # 实际上DQN内部会重新计算TD目标
                    )
                    
                    # 添加到缓冲区
                    buffers[agent_id].add(experience)
                
                # 更新观测
                obs_dict = next_obs_dict
                
                # 如果回合结束，重置环境
                if done:
                    obs_dict, _ = env.reset(seed=args.seed + epoch + i)
        
        # 收集数据和训练
        episode_returns = {agent_id: 0.0 for agent_id in env.agents}
        episode_steps = 0
        
        obs_dict, _ = env.reset(seed=args.seed + epoch * 100)
        
        for i in range(args.step_per_epoch):
            # 选择动作
            actions = {}
            for agent_id, policy in agents.items():
                # 转换观测为tensor
                obs_tensor = torch.tensor(
                    obs_dict[agent_id], 
                    dtype=torch.float32, 
                    device=device
                ).unsqueeze(0)
                
                # 选择动作
                result = policy(Batch(obs=obs_tensor, info={}))
                # 确保动作是int类型
                if torch.is_tensor(result.act):
                    actions[agent_id] = result.act.item()
                else:
                    actions[agent_id] = int(result.act)
            
            # 执行动作
            next_obs_dict, rewards, terminations, truncations, infos, done = env.step(actions)
            
            # 更新累积奖励
            for agent_id in env.agents:
                episode_returns[agent_id] += rewards[agent_id]
            
            episode_steps += 1
            
            # 为每个智能体存储经验
            for agent_id in env.agents:
                # 计算n步return (简化版的计算)
                rew = rewards[agent_id]
                done_flag = terminations[agent_id] or truncations[agent_id]
                
                # 这里同样不直接计算returns
                # DQN策略内部会根据reward, done和下一状态计算TD目标
                
                # 创建经验
                experience = Batch(
                    obs=np.array([obs_dict[agent_id]], dtype=np.float32),
                    act=np.array([actions[agent_id]], dtype=np.int64),
                    rew=np.array([rew], dtype=np.float32),
                    terminated=np.array([terminations[agent_id]], dtype=np.bool_),
                    truncated=np.array([truncations[agent_id]], dtype=np.bool_),
                    done=np.array([done_flag], dtype=np.bool_),
                    obs_next=np.array([next_obs_dict[agent_id]], dtype=np.float32),
                    info=np.array([{}]),
                    # 提供returns字段，但实际上DQN内部会重新计算TD目标
                    returns=np.array([rew], dtype=np.float32)
                )
                
                # 添加到缓冲区
                buffers[agent_id].add(experience)
            
            # 更新观测
            obs_dict = next_obs_dict
            
            # 如果回合结束，记录回报并重置环境
            if done or episode_steps >= 300:
                for agent_id in env.agents:
                    epoch_returns[agent_id].append(episode_returns[agent_id])
                    # 记录每个回合的奖励
                    total_episodes = epoch * (args.step_per_epoch // 300) + (i // 300)
                    writer.add_scalar(f'train/{agent_id}/episode_reward', 
                                     episode_returns[agent_id], 
                                     global_step=total_episodes)
                
                print(f"训练回合结束，步数: {episode_steps}, 奖励: {episode_returns}")
                
                # 重置环境和回报
                obs_dict, _ = env.reset(seed=args.seed + epoch * 100 + i + 1)
                episode_returns = {agent_id: 0.0 for agent_id in env.agents}
                episode_steps = 0
            
            # 更新策略
            if i % args.update_freq == 0:
                for agent_id, policy in agents.items():
                    # 确保缓冲区有足够的样本
                    if len(buffers[agent_id]) >= args.batch_size:
                        for _ in range(max(1, int(args.update_per_step * args.update_freq))):
                            try:
                                # 从经验回放缓冲区采样
                                batch, indices = buffers[agent_id].sample(args.batch_size)
                                
                                # 确保batch包含必要字段
                                if not hasattr(batch, 'returns') or batch.returns is None:
                                    # 这里实际上DQN会内部计算TD目标
                                    # 我们只是为了满足tianshou的接口要求提供一个初始值
                                    batch.returns = batch.rew
                                
                                # 更新策略 - DQN内部会计算真正的Q值目标
                                losses = policy.learn(batch)
                                
                                # 记录损失 (确保losses有loss属性)
                                if hasattr(losses, 'loss'):
                                    writer.add_scalar(f'train/{agent_id}/loss', 
                                                    losses.loss,
                                                    global_step=epoch * args.step_per_epoch + i)
                            except Exception as e:
                                print(f"更新策略时出错: {e}")
        
        # 记录每个智能体的平均回报
        for agent_id in env.agents:
            if epoch_returns[agent_id]:
                avg_return = np.mean(epoch_returns[agent_id])
                writer.add_scalar(f'train/{agent_id}/epoch_reward', avg_return, global_step=epoch)
                print(f"{agent_id} - 平均奖励: {avg_return:.4f}")
        
        # 测试阶段
        print("\n测试...")
        for agent_id, policy in agents.items():
            policy.eval()
            policy.set_eps(args.eps_test)
        
        # 创建测试环境
        test_env = make_env_fn(world="21_05", use_pred=use_predator, render=args.render_test)()
        all_rewards = []
        
        for i in range(args.test_num):
            episode_rewards = {agent_id: 0.0 for agent_id in test_env.agents}
            obs_dict, _ = test_env.reset(seed=args.seed + 10000 + i)
            done = False
            step = 0
            
            while not done and step < 300:
                actions = {}
                for agent_id, policy in agents.items():
                    # 提取观测
                    agent_obs = obs_dict[agent_id]
                    # 转换为张量
                    obs_tensor = torch.tensor(
                        agent_obs, 
                        dtype=torch.float32, 
                        device=device
                    ).unsqueeze(0)
                    
                    with torch.no_grad():
                        # 选择动作
                        result = policy(Batch(obs=obs_tensor, info={}))
                        # 提取动作
                        if torch.is_tensor(result.act):
                            action = result.act.item()
                        else:
                            action = int(result.act)
                    
                    actions[agent_id] = action
                
                # 执行动作
                next_obs_dict, rewards, terminations, truncations, infos, done = test_env.step(actions)
                print(f"测试回合 {i+1}/{args.test_num}, 步数: {step}, 奖励: {rewards}")
                
                # 更新奖励
                for agent_id in test_env.agents:
                    episode_rewards[agent_id] += rewards[agent_id]
                
                # 更新观测和步数
                obs_dict = next_obs_dict
                step += 1
                
                # 渲染
                if args.render_test:
                    test_env.render()
                    time.sleep(0.01)
            
            # 检查任务完成情况
            goal_reached = {}
            captured = {}
            for agent_id in test_env.agents:
                # 检查是否达到目标
                goal_reached[agent_id] = 0
                captured[agent_id] = 0
                if agent_id in infos:
                    if "is_success" in infos[agent_id] and infos[agent_id]["is_success"]:
                        goal_reached[agent_id] = 1
                    if "captures" in infos[agent_id] and infos[agent_id]["captures"] > 0:
                        captured[agent_id] = 1
            
            all_rewards.append([episode_rewards[agent_id] for agent_id in test_env.agents])
            print(f"测试回合 {i+1}/{args.test_num}, 步数: {step}, 奖励: {episode_rewards}")
            print(f"目标达成: {goal_reached}, 被捕获: {captured}")
        
        # 计算平均奖励
        all_rewards = np.array(all_rewards)
        avg_rewards = all_rewards.mean(axis=0)
        for i, agent_id in enumerate(test_env.agents):
            writer.add_scalar(f'test/{agent_id}/reward', avg_rewards[i], global_step=epoch)
            print(f"{agent_id} 平均奖励: {avg_rewards[i]:.4f}")
        
        # 保存模型
        if epoch == args.epoch - 1 or epoch % 5 == 0:
            for agent_id, policy in agents.items():
                torch.save(policy.state_dict(), os.path.join(log_path, f"{agent_id}_epoch{epoch}.pth"))
    
    # 训练结束，保存最终模型
    print("\n训练完成! 保存最终模型...")
    for agent_id, policy in agents.items():
        torch.save(policy.state_dict(), os.path.join(log_path, f"{agent_id}_final.pth"))
    
    writer.close()
    # 返回训练好的智能体
    return agents, log_path

def test_dqn_agents(args, agents=None):
    """测试训练好的DQN智能体"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    use_predator = not getattr(args, 'no_predator', False)
    env = make_env_fn(world="21_05", use_pred=use_predator, render=args.render_test)()
    print(f"创建环境成功，智能体列表: {env.agents}")
    
    # 如果没有提供训练好的智能体，则尝试加载
    if agents is None:
        agents = {}
        
        # 获取观测和动作空间
        obs_dict, _ = env.reset(seed=args.seed)
        
        for agent_id in env.agents:
            # 获取观测维度和动作维度
            obs_shape = len(obs_dict[agent_id])
            action_dim = env.action_spaces[agent_id].n
            
            # 创建网络
            net = make_dqn_net(
                state_dim=obs_shape, 
                action_dim=action_dim,
                device=device,
                hidden_sizes=args.hidden_sizes
            )
            
            # 创建DQN策略
            policy = DQNPolicy(
                model=net,
                optim=torch.optim.Adam(net.parameters(), lr=args.lr),
                discount_factor=args.gamma,
                estimation_step=args.n_step,
                target_update_freq=args.target_update_freq,
                action_space=env.action_spaces[agent_id],
                is_double=True
            )
            
            # 尝试加载模型
            model_loaded = False
            for model_type in ["final", "best"]:
                model_path = os.path.join(args.save_path, f"{agent_id}_{model_type}.pth")
                try:
                    policy.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"加载 {agent_id} 的{model_type}模型: {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"未找到或无法加载 {model_path}: {e}")
            
            if not model_loaded:
                print(f"警告: {agent_id} 未找到训练好的模型，将使用随机策略!")
            
            agents[agent_id] = policy
    
    # 设置为评估模式
    for agent_id, policy in agents.items():
        policy.eval()
        policy.set_eps(args.eps_test)
    
    # 运行测试回合
    all_rewards = []
    goal_reached = {agent_id: 0 for agent_id in env.agents}
    captured = {agent_id: 0 for agent_id in env.agents}
    
    for i in range(args.test_num):
        episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
        obs_dict, _ = env.reset(seed=args.seed + 20000 + i)
        done = False
        step = 0
        
        while not done and step < 300:
            actions = {}
            for agent_id, policy in agents.items():
                # 提取观测
                agent_obs = obs_dict[agent_id]
                # 转换为张量
                obs_tensor = torch.tensor(
                    agent_obs, 
                    dtype=torch.float32, 
                    device=device
                ).unsqueeze(0)
                
                with torch.no_grad():
                    # 选择动作
                    result = policy(Batch(obs=obs_tensor, info={}))
                    # 提取动作
                    if torch.is_tensor(result.act):
                        action = result.act.item()
                    else:
                        action = int(result.act)
                
                actions[agent_id] = action
            
            # 执行动作
            next_obs_dict, rewards, terminations, truncations, infos, done = env.step(actions)
            
            # 更新奖励
            for agent_id in env.agents:
                episode_rewards[agent_id] += rewards[agent_id]
            
            # 更新观测和步数
            obs_dict = next_obs_dict
            step += 1
            
            # 渲染
            if args.render_test:
                env.render()
                time.sleep(0.01)
        
        # 检查任务完成情况
        for agent_id in env.agents:
            # 检查是否达到目标
            if agent_id in infos:
                if "is_success" in infos[agent_id] and infos[agent_id]["is_success"]:
                    goal_reached[agent_id] += 1
                if "captures" in infos[agent_id] and infos[agent_id]["captures"] > 0:
                    captured[agent_id] += 1
        
        all_rewards.append([episode_rewards[agent_id] for agent_id in env.agents])
        print(f"测试回合 {i+1}/{args.test_num}, 步数: {step}, 奖励: {episode_rewards}")
    
    # 计算平均奖励
    all_rewards = np.array(all_rewards)
    avg_rewards = all_rewards.mean(axis=0)
    print("\n测试结果:")
    for i, agent_id in enumerate(env.agents):
        print(f"{agent_id} 平均奖励: {avg_rewards[i]:.4f}")
    
    # 打印目标达成和被捕获统计
    print("\n目标达成次数:")
    for agent_id in env.agents:
        success_rate = goal_reached[agent_id] / args.test_num * 100
        print(f"  {agent_id}: {goal_reached[agent_id]}/{args.test_num} ({success_rate:.1f}%)")
    
    print("\n被捕获次数:")
    for agent_id in env.agents:
        capture_rate = captured[agent_id] / args.test_num * 100
        print(f"  {agent_id}: {captured[agent_id]}/{args.test_num} ({capture_rate:.1f}%)")
    
    return all_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--update-freq", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--buffer-size", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--eps-test", type=float, default=0.05)
    # parser.add_argument("--reward-threshold", type=float, default=0.9)
    parser.add_argument("--no-predator", action="store_true", help="禁用捕食者")
    parser.add_argument("--logdir", type=str, default="./log")
    parser.add_argument("--save-path", type=str, default="./log/dqn/policy")
    parser.add_argument("--render-test", action="store_true", help="测试时渲染环境")
    parser.add_argument("--test-only", action="store_true", help="仅测试不训练")

    # 如果是在IPython或Jupyter环境中
    try:
        import sys
        if "ipykernel" in sys.modules:
            # Jupyter环境下设置默认参数，以方便快速测试
            args = parser.parse_args([
                "--epoch", "2", 
                "--step-per-epoch", "500",
                "--test-num", "2",
                "--render-test"
            ])
        else:
            args = parser.parse_args()
    except:
        args = parser.parse_args()
    
    # 如果是测试模式
    if args.test_only:
        test_dqn_agents(args)
    else:
        # 训练并测试
        agents, log_path = train_dqn_agents(args)
        # 更新保存路径
        args.save_path = log_path
        # 测试
        test_dqn_agents(args, agents) 