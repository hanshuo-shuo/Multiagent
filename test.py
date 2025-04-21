import time
from env import DualEvadeEnv

if __name__ == "__main__":
    def reward(obs):
        return 1
    env = DualEvadeEnv(
                   world_name= "21_05",
                   use_lppos=False,
                   use_predator=True,
                   reward_function=reward,
                   render=True,
                   real_time=True,
                   use_other=True)
    env.reset()
    start_time = time.time()
    step_count = 0

    for i in range(100):
        action = env.action_space.sample()
        for j in range(10):
            env.step(action=action)
            step_count += 1

    total_time = time.time() - start_time

    print("steps per second: {:.2f}".format(step_count / total_time))
# from n_evade_env import MultiEvadeEnv
# # 创建一个包含3个智能体的环境
# env = MultiEvadeEnv(
#     world_name="21_05",
#     n_agents=2,
#     use_predator=True,
#     render=True,
#     real_time=True
# )

# # 设置自定义策略（如果需要）
# def custom_policy(obs):
#     # 实现你的策略
#     return env.action_space.sample()

# # 为第二个智能体设置策略
# env.set_other_policy(0, custom_policy)

# # 运行环境
# obs, _ = env.reset()
# for _ in range(100):
#     action = env.action_space.sample()  # 控制第一个智能体的动作
#     print(action)
#     obs, reward, done, truncated, info = env.step(action)
#     if done:
#         obs, _ = env.reset()
