import collections
import random
import os
from datetime import datetime
import numpy as np
from env import DualEvadeEnv
from reward import custom_reward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from pz_env import env as make_env

# Optimized Hyperparameters
learning_rate = 0.0001  # Optimized learning rate
gamma = 0.99  # Higher discount factor for better long-term planning
buffer_limit = 200000  # Larger replay buffer
batch_size = 256  # Larger batch size for more stable gradients
log_interval = 10  # More frequent logging
max_episodes = 1000
min_buffer_size = 1000  # Minimum buffer size before training starts
soft_update_tau = 0.005  # For soft updates instead of hard updates

# Exploration parameters
initial_epsilon = 1
final_epsilon = 0.01
epsilon_decay = 0.99  # Exponential decay

# Environment parameters
max_steps = 300
render = False  # Set to True to visualize training

# Early stopping parameters
patience = 50  # Number of episodes to wait for improvement
min_improvement = 0.01  # Minimum improvement to reset patience counter

# Network structure parameters
hidden_size = 256  # Larger hidden size
num_layers = 3  # More layers for a deeper network

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        
        # 使用numpy.array转换列表为数组，避免性能警告
        s_array = np.array(s_lst, dtype=np.float32)
        a_array = np.array(a_lst)
        r_array = np.array(r_lst)
        s_prime_array = np.array(s_prime_lst, dtype=np.float32)
        done_mask_array = np.array(done_mask_lst)

        return torch.tensor(s_array, dtype=torch.float), torch.tensor(a_array), \
               torch.tensor(r_array), torch.tensor(s_prime_array, dtype=torch.float), \
               torch.tensor(done_mask_array)
    
    def size(self):
        return len(self.buffer)

class DuelingQnet(nn.Module):
    """
    Improved Dueling DQN architecture with:
    - More layers
    - Larger hidden layers
    - Dueling Q-network structure
    - Layer Normalization
    """
    def __init__(self, obs_dim, action_dim):
        super(DuelingQnet, self).__init__()
        # Feature extraction layers
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim)
        )
        
    def forward(self, x):
        features = self.layers(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 修复维度问题：如果输入是单个样本，则没有批次维度
        if advantages.dim() == 1:
            # 单样本情况下，直接使用所有动作的平均值
            return value + (advantages - advantages.mean())
        else:
            # 批量样本情况下，对每个样本的动作进行平均
            return value + (advantages - advantages.mean(dim=1, keepdim=True))
      
    def sample_action(self, obs, epsilon):
        # Convert numpy array to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        # 确保输入有正确的维度
        if obs.dim() == 1:
            # 如果是单个样本，添加批次维度
            obs = obs.unsqueeze(0)
        
        # Use epsilon-greedy policy for exploration
        if random.random() < epsilon:
            return random.randint(0, self.advantage_stream[-1].out_features - 1)
        else:
            with torch.no_grad():
                # 确保正确获取argmax并返回标量
                return self.forward(obs).squeeze(0).argmax().item()

def train(q, q_target, memory, optimizer):
    if memory.size() < batch_size:
        return 0  # Not enough samples
    
    total_loss = 0
    # Perform multiple training steps
    for _ in range(10):  # Increased number of training iterations per episode
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        
        # 确保所有张量都在正确的设备上
        device = next(q.parameters()).device
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_prime = s_prime.to(device)
        done_mask = done_mask.to(device)

        # Get current Q values
        q_out = q(s)
        q_a = q_out.gather(1, a)
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_q_values = q(s_prime)
            best_actions = next_q_values.argmax(1, keepdim=True)
            next_q_target_values = q_target(s_prime)
            max_q_prime = next_q_target_values.gather(1, best_actions)
            
            # Calculate target Q values
            target = r + gamma * max_q_prime * done_mask
        
        # Huber loss for stability
        loss = F.smooth_l1_loss(q_a, target)
        total_loss += loss.item()
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=10)
        optimizer.step()
    
    return total_loss / 8  # Return average loss

def soft_update(target, source, tau):
    """
    Soft update model parameters.
    θ_target = τ*θ_source + (1 - τ)*θ_target
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )

def plot_rewards(rewards_history, log_dir, window=100):
    """
    Plot the reward curves for all agents
    """
    plt.figure(figsize=(12, 8))
    
    has_data_to_plot = False
    for agent_name, rewards in rewards_history.items():
        # Calculate moving average with specified window
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = range(window-1, len(rewards))
            plt.plot(x, moving_avg, label=f"{agent_name} ({len(rewards)} episodes)")
            has_data_to_plot = True
        elif len(rewards) > 0:
            plt.plot(rewards, label=f"{agent_name} ({len(rewards)} episodes)")
            has_data_to_plot = True
    
    plt.title("Training Rewards Moving Average")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    
    # 只有当有数据绘制时才添加legend
    if has_data_to_plot:
        plt.legend()
    
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(log_dir, "reward_curves.png"))
    plt.close()

def main():
    # Create log directory
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('log', f'dqn_multiagent_{current_time}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup device - use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")
    
    # Create environment
    env = make_env(
        world_name="21_05",
        use_lppos=True,
        use_predator=True,
        max_step=300,
        reward_function=custom_reward,
        time_step=0.25,
        render=render,
        real_time=render,  # Only use real-time when rendering
        end_on_pov_goal=False,
        use_other=True,
        action_type=DualEvadeEnv.ActionType.DISCRETE
    )
    
    # Get observation and action dimensions for the first reset
    observations, _ = env.reset()
    
    # Create agents (one per prey)
    agents = {}
    rewards_history = {}  # Store rewards for plotting
    best_rewards = {}  # Track best rewards for early stopping
    patience_counter = {}  # Track patience for early stopping
    
    for agent_name in env.possible_agents:
        obs_dim = env.observation_spaces[agent_name].shape[0]
        
        if isinstance(env.action_spaces[agent_name], gym.spaces.Discrete):
            act_dim = env.action_spaces[agent_name].n
        else:
            raise ValueError("This implementation only supports discrete action spaces")
        
        # Create Q-networks and target networks with improved architecture
        q_net = DuelingQnet(obs_dim, act_dim).to(device)
        q_target = DuelingQnet(obs_dim, act_dim).to(device)
        q_target.load_state_dict(q_net.state_dict())
        
        # Create optimizer with learning rate scheduler
        optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
        memory = ReplayBuffer()
        
        agents[agent_name] = {
            "q_net": q_net,
            "q_target": q_target,
            "optimizer": optimizer,
            "memory": memory,
            "score": 0.0,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "training_losses": []  # Track losses
        }
        
        rewards_history[agent_name] = []
        best_rewards[agent_name] = -float('inf')
        patience_counter[agent_name] = 0
    
    # Log all hyperparameters
    with open(os.path.join(log_dir, "parameters.txt"), "w") as f:
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"buffer_limit: {buffer_limit}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"hidden_size: {hidden_size}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"soft_update_tau: {soft_update_tau}\n")
        f.write(f"initial_epsilon: {initial_epsilon}\n")
        f.write(f"final_epsilon: {final_epsilon}\n")
        f.write(f"epsilon_decay: {epsilon_decay}\n")
    
    # Create progress bar
    pbar = tqdm(range(max_episodes), desc="Training")
    
    # Initialize epsilon
    epsilon = initial_epsilon
    
    # Training loop
    for episode in pbar:
        # Reset environment
        observations, _ = env.reset()
        
        # Episode loop
        done = False
        episode_rewards = {agent: 0.0 for agent in env.agents}
        
        while not done:
            # Each agent selects action based on current observation
            actions = {}
            for agent_name in env.agents:
                obs = observations[agent_name]
                # Move observation to device
                obs_tensor = torch.tensor(obs, dtype=torch.float).to(device)
                action = agents[agent_name]["q_net"].sample_action(obs_tensor, epsilon)
                actions[agent_name] = action
            
            # Execute actions in environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store transitions in replay buffers and update observations
            for agent_name in env.agents:
                obs = observations[agent_name]
                action = actions[agent_name]
                reward = rewards[agent_name]
                next_obs = next_observations[agent_name]
                
                # Check if episode is done for this agent
                terminated = terminations[agent_name]
                truncated = truncations[agent_name]
                done_mask = 0.0 if (terminated or truncated) else 1.0
                
                # Store transition
                agents[agent_name]["memory"].put((obs, action, reward, next_obs, done_mask))
                
                # Update agent score
                agents[agent_name]["score"] += reward
                episode_rewards[agent_name] += reward
            
            # Update observations
            observations = next_observations
            
            # Check if episode is done
            done = all(terminations.values()) or all(truncations.values())
        
        # Decay epsilon
        epsilon = max(final_epsilon, epsilon * epsilon_decay)
        
        # Train agents if enough samples are collected
        for agent_name in env.agents:
            agent = agents[agent_name]
            
            # Append episode reward to history
            rewards_history[agent_name].append(episode_rewards[agent_name])
            
            # Check if there's enough data to start training
            if agent["memory"].size() > min_buffer_size:
                avg_loss = train(agent["q_net"], agent["q_target"], agent["memory"], agent["optimizer"])
                agent["training_losses"].append(avg_loss)
            
                # Soft update target network
                soft_update(agent["q_target"], agent["q_net"], soft_update_tau)
        
        # Log training progress
        if episode % log_interval == 0:
            log_msg = f"Episode: {episode:5d} | Epsilon: {epsilon:.3f}"
            for agent_name in env.agents:
                # Update win rate over the last 100 episodes
                recent_rewards = rewards_history[agent_name][-100:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                
                # Update progress bar
                log_msg += f" | {agent_name} - AvgR: {avg_reward:.3f}"
                
                # Early stopping check
                if avg_reward > best_rewards[agent_name] + min_improvement:
                    best_rewards[agent_name] = avg_reward
                    patience_counter[agent_name] = 0
                    
                    # Save best model
                    best_model_path = os.path.join(log_dir, f"{agent_name}_best.pt")
                    torch.save(agents[agent_name]["q_net"].state_dict(), best_model_path)
                else:
                    patience_counter[agent_name] += 1
            
            # Update progress bar
            pbar.set_postfix_str(log_msg)
            
            # Plot and save reward curves periodically
            if episode % (log_interval * 10) == 0:
                plot_rewards(rewards_history, log_dir)
                
                # Save loss curves
                plt.figure(figsize=(12, 8))
                has_loss_to_plot = False
                for agent_name in env.agents:
                    losses = agents[agent_name]["training_losses"]
                    if losses:
                        # Smooth losses for better visualization
                        window_size = min(50, len(losses) // 10 + 1)
                        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                        plt.plot(smoothed_losses, label=f"{agent_name} Loss")
                        has_loss_to_plot = True
                
                plt.title("Training Loss")
                plt.xlabel("Training Steps")
                plt.ylabel("Loss")
                if has_loss_to_plot:
                    plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(log_dir, "loss_curves.png"))
                plt.close()
            
            # Check early stopping
            should_stop = True
            for agent_name in env.agents:
                if patience_counter[agent_name] < patience:
                    should_stop = False
                    break
            
            if should_stop and episode > min_buffer_size:
                print(f"Early stopping at episode {episode} due to no improvement")
                break
        
        # Regularly save checkpoints
        if episode % (log_interval * 20) == 0 and episode > 0:
            for agent_name in env.agents:
                checkpoint_path = os.path.join(log_dir, f"{agent_name}_ep{episode}.pt")
                torch.save(agents[agent_name]["q_net"].state_dict(), checkpoint_path)
    
    # Close environment
    env.close()
    
    # Save final models
    for agent_name in env.agents:
        model_path = os.path.join(log_dir, f"{agent_name}_final.pt")
        torch.save(agents[agent_name]["q_net"].state_dict(), model_path)
    
    # Plot final reward curves
    plot_rewards(rewards_history, log_dir)
    
    print("\nTraining complete!")
    print(f"Logs and models saved to: {log_dir}")

def evaluate(model_paths, num_episodes=10, render=True):
    """
    评估已训练好的智能体模型
    
    参数:
        model_paths: 字典，键为智能体名称，值为对应模型路径
        num_episodes: 评估的回合数
        render: 是否渲染环境
    
    返回:
        评估结果统计信息
    """
    print(f"开始评估 {num_episodes} 个回合...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = make_env(
        world_name="21_05",
        use_lppos=True,
        use_predator=True,
        max_step=300,
        reward_function=custom_reward,
        time_step=0.25,
        render=render,
        real_time=render,
        end_on_pov_goal=False,
        use_other=True,
        action_type=DualEvadeEnv.ActionType.DISCRETE
    )
    
    # 获取初始观察和动作空间维度
    observations, _ = env.reset()
    
    # 初始化智能体模型
    agents = {}
    for agent_name in env.possible_agents:
        obs_dim = env.observation_spaces[agent_name].shape[0]
        
        if isinstance(env.action_spaces[agent_name], gym.spaces.Discrete):
            act_dim = env.action_spaces[agent_name].n
        else:
            raise ValueError("只支持离散动作空间")
        
        # 创建模型
        q_net = DuelingQnet(obs_dim, act_dim).to(device)
        
        # 加载模型权重
        if agent_name in model_paths:
            q_net.load_state_dict(torch.load(model_paths[agent_name], map_location=device))
            print(f"已加载 {agent_name} 的模型: {model_paths[agent_name]}")
        else:
            print(f"警告: 未找到 {agent_name} 的模型路径")
        
        # 设置为评估模式
        q_net.eval()
        
        agents[agent_name] = {
            "q_net": q_net,
            "obs_dim": obs_dim,
            "act_dim": act_dim
        }
    
    # 统计数据
    rewards_history = {agent_name: [] for agent_name in env.possible_agents}
    episode_lengths = []
    success_rates = {agent_name: 0 for agent_name in env.possible_agents}
    capture_counts = {agent_name: 0 for agent_name in env.possible_agents}
    
    # 评估循环
    for episode in range(num_episodes):
        # 重置环境
        observations, _ = env.reset()
        
        # 记录单个回合信息
        episode_rewards = {agent_name: 0.0 for agent_name in env.agents}
        step_count = 0
        done = False
        
        # 回合循环
        while not done:
            # 记录步数
            step_count += 1
            
            # 每个智能体选择动作
            actions = {}
            for agent_name in env.agents:
                obs = observations[agent_name]
                # 移动观察到设备
                obs_tensor = torch.tensor(obs, dtype=torch.float).to(device)
                
                # 使用Q网络选择最优动作（无探索）
                with torch.no_grad():
                    if obs_tensor.dim() == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    q_values = agents[agent_name]["q_net"](obs_tensor)
                    action = q_values.argmax().item()
                    actions[agent_name] = action
            
            # 执行动作
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 更新回合奖励
            for agent_name in env.agents:
                episode_rewards[agent_name] += rewards[agent_name]
            
            # 更新观察
            observations = next_observations
            
            # 检查回合是否结束
            done = all(terminations.values()) or all(truncations.values())
        
        # 记录结果
        episode_lengths.append(step_count)
        
        # 收集每个智能体的统计信息
        for agent_name in env.agents:
            rewards_history[agent_name].append(episode_rewards[agent_name])
            
            # 检查成功和被捕获信息
            if agent_name in infos and "is_success" in infos[agent_name]:
                if infos[agent_name]["is_success"]:
                    success_rates[agent_name] += 1
            
            if agent_name in infos and "captures" in infos[agent_name]:
                if infos[agent_name]["captures"] > 0:
                    capture_counts[agent_name] += 1
        
        # 打印单回合结果
        print(f"回合 {episode+1}/{num_episodes} - 步数: {step_count}")
        for agent_name in env.agents:
            print(f"  {agent_name} - 奖励: {episode_rewards[agent_name]:.4f}")
    
    # 计算总体统计信息
    print("\n===== 评估结果 =====")
    print(f"总回合数: {num_episodes}")
    print(f"平均回合长度: {sum(episode_lengths)/len(episode_lengths):.2f} 步")
    
    for agent_name in env.possible_agents:
        avg_reward = sum(rewards_history[agent_name]) / num_episodes
        success_rate = success_rates[agent_name] / num_episodes * 100
        capture_rate = capture_counts[agent_name] / num_episodes * 100
        
        print(f"\n{agent_name} 统计:")
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  成功率: {success_rate:.2f}%")
        print(f"  被捕获率: {capture_rate:.2f}%")
    
    # 关闭环境
    env.close()
    
    return {
        "rewards": rewards_history,
        "episode_lengths": episode_lengths,
        "success_rates": success_rates,
        "capture_counts": capture_counts
    }

# 示例用法
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN训练或评估')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='运行模式: train (训练) 或 eval (评估)')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='评估模式下，模型所在文件夹路径')
    parser.add_argument('--episodes', type=int, default=10,
                        help='评估模式下，评估的回合数')
    parser.add_argument('--render', action='store_true',
                        help='是否渲染环境')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        main()
    elif args.mode == 'eval':
        if args.model_dir:
            # 构建模型路径
            model_paths = {}
            for agent in ["prey_1", "prey_2"]:
                # 尝试读取 best, final 或特定回合的模型
                best_path = os.path.join(args.model_dir, f"{agent}_bes.pt")
                final_path = os.path.join(args.model_dir, f"{agent}_final.pt")
                
                if os.path.exists(best_path):
                    model_paths[agent] = best_path
                elif os.path.exists(final_path):
                    model_paths[agent] = final_path
                else:
                    # 尝试找到最新的checkpoint
                    agent_models = [f for f in os.listdir(args.model_dir) if f.startswith(agent) and f.endswith('.pt')]
                    if agent_models:
                        model_paths[agent] = os.path.join(args.model_dir, agent_models[-1])
                    else:
                        print(f"错误: 未找到 {agent} 的模型")
            
            if model_paths:
                evaluate(model_paths, num_episodes=args.episodes, render=args.render)
            else:
                print("错误: 未找到模型文件")
        else:
            print("错误: 评估模式需要指定模型目录 (--model_dir)")
