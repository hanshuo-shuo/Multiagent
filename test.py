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

    total_time = time.time() - start_time

    print("steps per second: {:.2f}".format(step_count / total_time))