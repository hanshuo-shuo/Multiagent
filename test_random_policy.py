from pz_env import env, random_policy
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

if __name__ == "__main__":
    # 创建多智能体环境
    env = env(
        world_name="21_05",
        use_lppos=True,
        use_predator=True,
        max_step=300,
        reward_function=custom_reward,
        time_step=0.25,
        render=True,
        real_time=True,
        end_on_pov_goal=True,
        use_other=True,
        action_type=DualEvadeEnv.ActionType.CONTINUOUS  # 可以改为DISCRETE测试离散动作
    )
    
    print("环境信息:")
    print(f"观测空间: {env.observation_spaces['prey_1']}")
    print(f"动作空间: {env.action_spaces['prey_1']}")
    print(f"智能体列表: {env.possible_agents}")
    
    # 运行随机策略
    try:
        rewards, success = random_policy(
            env=env,
            num_episodes=3,
            max_steps=300,
            render=True
        )
    finally:
        # 确保环境被关闭
        env.close() 