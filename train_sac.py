import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections
import random
import os
import argparse
import matplotlib.pyplot as plt
from pz_env import MultiDualEvadeEnv
from env import DualEvadeEnv
from reward import custom_reward

# Hyperparameters
lr_pi = 0.0003
lr_q = 0.0003
init_alpha = 1
gamma = 0.99
batch_size = 256
buffer_limit = 1000000
tau = 0.005  # for target network soft update (slower updates for better stability)
target_entropy = -2.0  # for automated alpha update (depends on action space dimension)
lr_alpha = 0.0003  # for automated alpha update
train_interval = 1  # 每个智能体每隔多少步进行一次训练
update_iterations = 1  # 每次训练更新多少次
num_gradients_per_step = 1  # 每步环境交互执行的梯度更新次数
start_training = 1000  # 开始训练前需要收集的经验数量
print_interval = 10  # 每多少个episode打印一次结果
max_episodes = 1000  # 最大训练回合数
max_steps = 300  # 每个回合最大步数
save_interval = 100  # 每多少个episode保存一次模型
model_path = "log/sac"  # 模型保存路径
hidden_dim = 256  # 隐藏层维度
reward_scale = 1.0  # 奖励缩放系数，用于调整熵正则化的影响

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{'GPU is' if torch.cuda.is_available() else 'GPU is not'} available, using {'GPU' if torch.cuda.is_available() else 'CPU'} instead")

def weights_init_(m, gain=1):
    """
    权重初始化函数，使用Xavier/Glorot初始化
    
    参数:
    - m: 网络模块
    - gain: 增益系数，控制初始化的方差
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        # 先转换为numpy数组再创建张量，以提高性能
        s_array = np.array(s_lst, dtype=np.float32)
        a_array = np.array(a_lst, dtype=np.float32)
        r_array = np.array(r_lst, dtype=np.float32)
        s_prime_array = np.array(s_prime_lst, dtype=np.float32)
        done_mask_array = np.array(done_mask_lst, dtype=np.float32)
        
        return torch.tensor(s_array).to(device), \
               torch.tensor(a_array).to(device), \
               torch.tensor(r_array).to(device), \
               torch.tensor(s_prime_array).to(device), \
               torch.tensor(done_mask_array).to(device)
    
    def size(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_pi)
        
        # 初始化网络权重
        self.apply(lambda m: weights_init_(m, gain=np.sqrt(2)))  # 隐藏层使用sqrt(2)增益
        # mu层使用较小的增益以产生更小的初始动作
        weights_init_(self.fc_mu, gain=0.01)  
        # std层也使用较小的增益以产生合理的初始方差
        weights_init_(self.fc_std, gain=0.01)
        
        # For automated entropy tuning
        self.action_bound = action_bound
        self.log_alpha = torch.tensor(np.log(init_alpha)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = target_entropy

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 1e-3  # To avoid std being too small
        
        # Create a Normal distribution with the parameterized mean and std
        dist = Normal(mu, std)
        
        # Sample using reparameterization trick
        action_0 = dist.rsample()
        
        # Calculate log probability
        log_prob = dist.log_prob(action_0).sum(dim=-1, keepdim=True)
        
        # Apply tanh squashing to confine action to [-1, 1]
        action = torch.tanh(action_0)
        
        # Correct log probability for the squashing
        log_prob = log_prob - torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1, keepdim=True)
        
        # Scale action to desired range
        action = action * self.action_bound
        
        return action, log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -(min_q + entropy).mean()  # Policy loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update temperature parameter alpha
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        return entropy.mean().item()

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_dim, hidden_dim)
        self.fc_a = nn.Linear(action_dim, hidden_dim)
        self.fc_cat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_q)
        
        # 初始化网络权重
        # 状态和动作编码层使用sqrt(2)增益
        weights_init_(self.fc_s, gain=np.sqrt(2))
        weights_init_(self.fc_a, gain=np.sqrt(2))
        # 合并层使用sqrt(2)增益
        weights_init_(self.fc_cat, gain=np.sqrt(2))
        # 输出层使用默认增益
        weights_init_(self.fc_out, gain=1.0)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target

def save_models(models, agent_name, episode):
    """保存模型"""
    if not os.path.exists(f"{model_path}/{agent_name}"):
        os.makedirs(f"{model_path}/{agent_name}")
    
    pi, q1, q2, q1_target, q2_target = models
    
    torch.save(pi.state_dict(), f"{model_path}/{agent_name}/pi_{episode}.pt")
    torch.save(q1.state_dict(), f"{model_path}/{agent_name}/q1_{episode}.pt")
    torch.save(q2.state_dict(), f"{model_path}/{agent_name}/q2_{episode}.pt")
    torch.save(q1_target.state_dict(), f"{model_path}/{agent_name}/q1_target_{episode}.pt")
    torch.save(q2_target.state_dict(), f"{model_path}/{agent_name}/q2_target_{episode}.pt")

def load_models(agent_name, episode, state_dim, action_dim, action_bound):
    """加载模型"""
    pi = PolicyNet(state_dim, action_dim, action_bound).to(device)
    q1, q2 = QNet(state_dim, action_dim).to(device), QNet(state_dim, action_dim).to(device)
    q1_target, q2_target = QNet(state_dim, action_dim).to(device), QNet(state_dim, action_dim).to(device)
    
    pi.load_state_dict(torch.load(f"{model_path}/{agent_name}/pi_{episode}.pt"))
    q1.load_state_dict(torch.load(f"{model_path}/{agent_name}/q1_{episode}.pt"))
    q2.load_state_dict(torch.load(f"{model_path}/{agent_name}/q2_{episode}.pt"))
    q1_target.load_state_dict(torch.load(f"{model_path}/{agent_name}/q1_target_{episode}.pt"))
    q2_target.load_state_dict(torch.load(f"{model_path}/{agent_name}/q2_target_{episode}.pt"))
    
    return pi, q1, q2, q1_target, q2_target

def plot_training_results(scores, success_rates, save_path=None, window_size=5):
    """
    绘制训练结果图表，包括两个智能体的reward和success rate变化
    
    参数:
    - scores: 各智能体的奖励记录字典
    - success_rates: 各智能体的成功率记录字典
    - save_path: 图表保存路径，如不指定则只显示不保存
    - window_size: 移动平均窗口大小
    """
    # 获取智能体名称
    agent_names = list(scores.keys())
    
    # 创建包含两行两列的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 为每个智能体绘制reward图表和success rate图表
    for i, agent_name in enumerate(agent_names):
        agent_scores = scores[agent_name]
        agent_success_rates = success_rates[agent_name]
        
        # 绘制reward图表（左列）
        ax_reward = axes[i, 0]
        
        # 计算移动平均
        if len(agent_scores) > window_size:
            # 计算移动平均
            cumsum = np.cumsum(np.insert(agent_scores, 0, 0))
            smoothed_scores = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
            
            # 绘制原始数据（半透明）
            ax_reward.plot(np.arange(len(agent_scores)), agent_scores, 
                          alpha=0.3, color='C'+str(i), label=f'{agent_name} (raw)')
            
            # 绘制平滑后的数据
            ax_reward.plot(np.arange(window_size-1, len(agent_scores)), 
                          smoothed_scores, linewidth=2, color='C'+str(i), 
                          label=f'{agent_name} (smoothed)')
        else:
            ax_reward.plot(np.arange(len(agent_scores)), agent_scores, 
                          label=agent_name)
        
        ax_reward.set_title(f'{agent_name} Reward')
        ax_reward.set_xlabel('Episode')
        ax_reward.set_ylabel('Episode Reward')
        ax_reward.grid(True, linestyle='--', alpha=0.7)
        ax_reward.legend()
        
        # 绘制success rate图表（右列）
        ax_success = axes[i, 1]
        
        # 计算累积成功率
        cumulative_success = np.cumsum(agent_success_rates)
        episodes = np.arange(1, len(agent_success_rates) + 1)
        cumulative_success_rate = cumulative_success / episodes
        
        # 绘制累积成功率
        ax_success.plot(episodes, cumulative_success_rate, 
                       linewidth=2, color='C'+str(i), label=agent_name)
        
        # 如果有数据，显示最终成功率
        if len(cumulative_success_rate) > 0:
            final_success_rate = cumulative_success_rate[-1]
            ax_success.text(len(episodes), final_success_rate,
                          f' {final_success_rate:.2f}',
                          verticalalignment='center')
        
        ax_success.set_title(f'{agent_name} Success Rate')
        ax_success.set_xlabel('Episode')
        ax_success.set_ylabel('Success Rate')
        ax_success.set_ylim([0, 1.05])
        ax_success.grid(True, linestyle='--', alpha=0.7)
        ax_success.legend()
    
    # 调整子图布局
    plt.tight_layout()
    
    # 保存图表（如果指定了保存路径）
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Training results plot saved to: {save_path}")
    
    # 显示图表
    plt.show()

def evaluate_policy(env, agents, num_episodes=10, max_steps=300, render=False):
    """
    评估策略性能，使用确定性策略（均值而非采样）
    
    参数:
    - env: 评估环境
    - agents: 智能体模型字典
    - num_episodes: 评估回合数
    - max_steps: 每回合最大步数
    - render: 是否渲染环境
    
    返回:
    - avg_rewards: 各智能体的平均奖励
    - success_rates: 各智能体的成功率
    """
    avg_rewards = {agent_name: 0 for agent_name in agents.keys()}
    success_counts = {agent_name: 0 for agent_name in agents.keys()}
    
    for episode in range(num_episodes):
        observations, infos = env.reset()
        episode_rewards = {agent_name: 0 for agent_name in agents.keys()}
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            actions = {}
            
            # 使用确定性策略（均值而非采样）
            for agent_name in agents.keys():
                pi, *_ = agents[agent_name]
                obs = torch.tensor(observations[agent_name], dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # 获取均值动作（不使用采样）
                    mu = torch.tanh(pi.fc_mu(F.relu(pi.fc2(F.relu(pi.fc1(obs)))))) * pi.action_bound
                    actions[agent_name] = mu.squeeze().cpu().numpy()
            
            # 执行动作
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 更新统计
            for agent_name in agents.keys():
                episode_rewards[agent_name] += rewards[agent_name]
            
            # 更新观测
            observations = next_observations
            
            # 如果需要渲染
            if render:
                env.render()
            
            # 检查是否结束
            done = all(terminations.values()) or all(truncations.values())
            step += 1
        
        # 累计奖励和成功次数
        for agent_name in agents.keys():
            avg_rewards[agent_name] += episode_rewards[agent_name]
            if agent_name in infos and "is_success" in infos[agent_name] and infos[agent_name]["is_success"]:
                success_counts[agent_name] += 1
    
    # 计算平均值
    for agent_name in agents.keys():
        avg_rewards[agent_name] /= num_episodes
    
    success_rates = {agent_name: success_counts[agent_name] / num_episodes for agent_name in agents.keys()}
    
    return avg_rewards, success_rates

def main(args):
    
    # 创建环境
    env_kwargs = {
        'world_name': args.world_name,
        'use_lppos': args.use_lppos,
        'use_predator': args.use_predator,
        'max_step': args.max_step,
        'time_step': args.time_step,
        'render': args.render,
        'real_time': args.real_time,
        'end_on_pov_goal': args.end_on_pov_goal,
        'use_other': True,  # 必须为True才能使用双智能体
        'action_type': DualEvadeEnv.ActionType.CONTINUOUS,  # 使用连续动作空间
        'reward_function': custom_reward  # 设置奖励函数
    }
    
    env = MultiDualEvadeEnv(**env_kwargs)
    
    # 获取观测和动作空间的维度
    agent = env.agents[0]  # 所有智能体的观测和动作空间都相同
    obs_dim = env.observation_spaces[agent].shape[0]
    act_dim = env.action_spaces[agent].shape[0]
    act_bound = 1.0  # 动作取值范围在[0,1]
    
    # 如果模型路径不存在则创建
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # 根据模式选择操作
    if args.mode == 'train':
        # 训练模式：执行原来的训练流程
        
        # 创建经验回放缓冲区
        memory = {agent: ReplayBuffer() for agent in env.agents}
        
        # 为每个智能体创建SAC模型
        agents = {}
        for agent_name in env.agents:
            pi = PolicyNet(obs_dim, act_dim, act_bound).to(device)
            q1, q2 = QNet(obs_dim, act_dim).to(device), QNet(obs_dim, act_dim).to(device)
            q1_target, q2_target = QNet(obs_dim, act_dim).to(device), QNet(obs_dim, act_dim).to(device)
            
            # 初始化目标网络
            q1_target.load_state_dict(q1.state_dict())
            q2_target.load_state_dict(q2.state_dict())
            
            agents[agent_name] = (pi, q1, q2, q1_target, q2_target)
        
        # 统计指标
        scores = {agent: [] for agent in env.agents}
        success_rates = {agent: [] for agent in env.agents}
        episode_score = {agent: 0 for agent in env.agents}
        episode_success = {agent: 0 for agent in env.agents}
        
        total_step = 0
        gradient_steps = 0
        
        for n_epi in range(1, max_episodes+1):
            observations, infos = env.reset()
            
            # 重置每个回合的统计
            for agent in env.agents:
                episode_score[agent] = 0
            
            done = False
            step = 0
            
            while not done and step < max_steps:
                # 为每个智能体选择动作
                actions = {}
                
                for agent_name in env.agents:
                    pi, *_ = agents[agent_name]
                    obs = torch.tensor(observations[agent_name], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # 如果经验足够则使用策略，否则随机探索
                    if memory[agent_name].size() > start_training:
                        with torch.no_grad():
                            action, _ = pi(obs)
                            action = action.squeeze().cpu().detach().numpy()
                    else:
                        action = np.random.uniform(0, 1, size=act_dim)
                    
                    actions[agent_name] = action
                
                # 执行动作
                next_observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # 判断是否结束
                done = all(terminations.values()) or all(truncations.values())
                
                # 存储经验
                for agent_name in env.agents:
                    memory[agent_name].put((
                        observations[agent_name],
                        actions[agent_name],
                        rewards[agent_name] * reward_scale,  # 应用奖励缩放
                        next_observations[agent_name],
                        done
                    ))
                    
                    # 累计奖励
                    episode_score[agent_name] += rewards[agent_name]
                
                # 更新观测
                observations = next_observations
                
                # 训练智能体 - 改为按照环境步数和梯度步数比例来更新
                if memory[agent_name].size() > start_training:
                    if total_step % train_interval == 0:
                        for _ in range(num_gradients_per_step):
                            for agent_name in env.agents:
                                pi, q1, q2, q1_target, q2_target = agents[agent_name]
                                
                                for _ in range(update_iterations):
                                    mini_batch = memory[agent_name].sample(batch_size)
                                    
                                    # 更新Q网络
                                    td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                                    q1.train_net(td_target, mini_batch)
                                    q2.train_net(td_target, mini_batch)
                                    
                                    # 更新策略网络
                                    pi.train_net(q1, q2, mini_batch)
                                    
                                    # 软更新目标网络
                                    q1.soft_update(q1_target)
                                    q2.soft_update(q2_target)
                                    
                                    gradient_steps += 1
                
                total_step += 1
                step += 1
            
            # 回合结束，记录成功率
            for agent_name in env.agents:
                if agent_name in infos and "is_success" in infos[agent_name] and infos[agent_name]["is_success"]:
                    episode_success[agent_name] = 1
                else:
                    episode_success[agent_name] = 0
                
                scores[agent_name].append(episode_score[agent_name])
                success_rates[agent_name].append(episode_success[agent_name])
            
            # 打印训练进度
            if n_epi % print_interval == 0:
                for agent_name in env.agents:
                    pi, *_ = agents[agent_name]
                    avg_score = np.mean(scores[agent_name][-print_interval:])
                    avg_success = np.mean(success_rates[agent_name][-print_interval:])
                    
                    print(f"Episode {n_epi}, {agent_name}: Avg Score: {avg_score:.2f}, "
                        f"Success Rate: {avg_success:.2f}, Alpha: {pi.log_alpha.exp().item():.4f}, "
                        f"Gradient Steps: {gradient_steps}")
            
            # 保存模型
            if n_epi % save_interval == 0:
                for agent_name in env.agents:
                    save_models(agents[agent_name], agent_name, n_epi)
                
                # 保存训练进度信息
                progress_file = f"{model_path}/training_progress_{args.world_name}.npy"
                np.save(progress_file, {
                    'episode': n_epi,
                    'scores': scores,
                    'success_rates': success_rates
                })
        
        # 训练结束，绘制结果图表
        if args.plot_results:
            plot_file = f"{model_path}/training_results_{args.world_name}.png"
            plot_training_results(scores, success_rates, save_path=plot_file)
        
    else:
        # 评估模式：加载已有模型并评估
        print(f"Evaluation mode: Loading models from episode {args.load_episode}")
        
        # 加载训练好的模型
        agents = {}
        for agent_name in env.agents:
            pi, q1, q2, q1_target, q2_target = load_models(
                agent_name, args.load_episode, obs_dim, act_dim, act_bound
            )
            agents[agent_name] = (pi, q1, q2, q1_target, q2_target)
        
        # 执行评估
        avg_rewards, success_rates = evaluate_policy(
            env, agents, 
            num_episodes=args.eval_episodes, 
            max_steps=max_steps, 
            render=args.render
        )
        
        # 打印评估结果
        print("\n=== Evaluation Results ===")
        for agent_name in env.agents:
            print(f"{agent_name}: Avg Reward: {avg_rewards[agent_name]:.2f}, "
                  f"Success Rate: {success_rates[agent_name]:.2f}")
        print("=========================\n")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent SAC training for dual evade environment")
    
    parser.add_argument("--mode", type=str, choices=['train', 'eval'], default='train',
                        help="Mode: 'train' for training, 'eval' for evaluation")
    parser.add_argument("--load_episode", type=int, default=1000,
                        help="Which episode to load model from for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=50,
                        help="Number of episodes to run for evaluation")
    
    parser.add_argument("--world_name", type=str, default="21_05", help="World configuration")
    parser.add_argument("--use_lppos", action="store_true", help="Use limited action list")
    parser.add_argument("--use_predator", default=True, help="Use predator in environment")
    parser.add_argument("--max_step", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--time_step", type=float, default=0.25, help="Time step for environment")
    parser.add_argument("--render", default=False, help="Render environment")
    parser.add_argument("--real_time", default=False, help="Run in real time")
    parser.add_argument("--end_on_pov_goal", default=False, help="End episode when POV reaches goal")
    parser.add_argument("--plot_results", action="store_true", help="Plot training results after training")
    
    args = parser.parse_args()
    
    main(args) 
