from pz_env import env

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


# 创建多智能体环境
env = env(
    world_name="21_05",
    use_lppos=False,
    use_predator=True,
    max_step=300,
    reward_function=custom_reward,
    time_step=0.25,
    render=True,
    real_time=True,
    end_on_pov_goal=True,
    use_other=True
)

# 示例并行交互
observations, infos = env.reset()
print("Initial observations:")
for agent, obs in observations.items():
    print(f"{agent}: position ({obs.self_x:.4f}, {obs.self_y:.4f})")
    print(f"  Other agent visible: {obs.other_x != 0 or obs.other_y != 0}")
    if obs.other_x != 0 or obs.other_y != 0:
        print(f"  Other agent position: ({obs.other_x:.4f}, {obs.other_y:.4f})")
    print(f"  Predator visible: {obs.predator_x != 0 or obs.predator_y != 0}")
    if obs.predator_x != 0 or obs.predator_y != 0:
        print(f"  Predator position: ({obs.predator_x:.4f}, {obs.predator_y:.4f})")
    print()

step_count = 0
terminations = {agent: False for agent in env.agents}
truncations = {agent: False for agent in env.agents}

# Run until any termination/truncation condition is met
while not any(terminations.values()) and not any(truncations.values()) and step_count < 100:
    step_count += 1
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
    
    print(f"\nStep {step_count}:")
    for agent in env.agents:
        obs = observations[agent]
        print(f"{agent}: position ({obs.self_x:.4f}, {obs.self_y:.4f}), reward: {rewards[agent]}")
        if terminations[agent]:
            print(f"  {agent} terminated!")
        if truncations[agent]:
            print(f"  {agent} truncated!")
    
    if any(terminations.values()) or any(truncations.values()):
        print("\nEnvironment done!")
        break