from tianshou.policy import SACPolicy
from tianshou.utils.net.continuous import ActorProb, Critic
import torch
import torch.nn as nn

# 简单网络测试（1D state, 1D action）
obs_dim = 3
action_dim = 2
hidden_sizes = [64, 64]
device = "cuda" if torch.cuda.is_available() else "cpu"

# actor
actor_net = nn.Sequential(
    nn.Linear(obs_dim, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], action_dim * 2)  # mean + log_std
)
actor = ActorProb(actor_net, action_shape=(action_dim,), max_action=1.0, device=device)

# critics
critic1 = Critic(nn.Sequential(
    nn.Linear(obs_dim + action_dim, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], 1)
), device=device)
critic2 = Critic(nn.Sequential(
    nn.Linear(obs_dim + action_dim, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], 1)
), device=device)

# optimizer
optim = torch.optim.Adam(list(actor.parameters()) + list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4)

# build policy
policy = SACPolicy(
    actor=actor,
    actor_optim=optim,
    critic1=critic1,
    critic1_optim=optim,
    critic2=critic2,
    critic2_optim=optim,
    tau=0.005,
    gamma=0.99,
    alpha=0.2,
    action_space=None
)

print("✅ SACPolicy 构建成功！")
