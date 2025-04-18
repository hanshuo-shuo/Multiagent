from pz_env import env

# 创建多智能体环境
env = env(
    world_name="21_05",
    use_lppos=False,
    use_predator=True,
    max_step=300,
    reward_function=lambda obs: 1,
    time_step=0.25,
    render=True,
    real_time=True,
    end_on_pov_goal=True,
    use_other=True
)

# 示例并行交互
observations = env.reset()
dones = {agent: False for agent in env.agents}
while True:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, dones, infos = env.step(actions)
    print(observations)
    if dones["__all__"]:
        break