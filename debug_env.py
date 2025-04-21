import numpy as np
import time
import argparse
from env import DualEvadeEnv, DualEvadeObservation

# Define reward function
def custom_reward(obs: DualEvadeObservation) -> float:
    """
    Simple reward function for debugging:
    - Penalty for getting puffed
    - Reward for getting closer to goal
    - Big reward for reaching goal
    """
    if obs.puffed > 0:
        return -10.0
    
    if not obs.finished > 0:
        # Regular reward during episode
        goal_reward = -0.1 * obs.prey_goal_distance  # Negative because we want to minimize distance
        time_reward = 0.05  # Small reward for staying alive
        return goal_reward + time_reward
    else:
        # Episode is finished
        if obs.prey_goal_distance < 0.5:  # Reached goal
            return 20.0
        return 0.0  # Terminated but didn't reach goal

def run_episode(env, max_steps=300, verbose=True, sleep_time=0.01):
    """Run a single episode with the given environment."""
    obs, _ = env.reset()
    
    # Print observation structure
    if verbose:
        print("Initial observation:")
        if isinstance(obs, np.ndarray):
            print(f"Type: numpy.ndarray, Shape: {obs.shape}")
            # Map array indices to field names
            fields = DualEvadeObservation.fields
            for i, field in enumerate(fields):
                print(f"  {field}: {obs[i]:.4f}")
        else:
            print(f"Type: {type(obs)}")
            for field in DualEvadeObservation.fields:
                value = getattr(obs, field)
                print(f"  {field}: {value:.4f}")
    
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    # Run episode
    while not done and not truncated and steps < max_steps:
        # Sample action based on action space
        if hasattr(env.action_space, 'n'):  # Check for discrete space
            # Discrete action space
            action = np.random.randint(0, env.action_space.n)
            action_desc = f"Discrete action: {action}"
        else:
            # Continuous action space
            action = env.action_space.sample()
            action_desc = f"Continuous action: ({action[0]:.4f}, {action[1]:.4f})"
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Update stats
        total_reward += reward
        steps += 1
        
        # Log step information
        if verbose and steps % 10 == 0:
            print(f"Step {steps}: {action_desc}")
            print(f"  Reward: {reward:.4f}, Total: {total_reward:.4f}")
            print(f"  Position: ({next_obs[0]:.4f}, {next_obs[1]:.4f})")
            print(f"  Goal distance: {next_obs[9]:.4f}")
            if env.model.use_predator:
                print(f"  Predator visible: {next_obs[7] != 0.0 or next_obs[8] != 0.0}")
        
        # Small delay for visualization
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Update observation
        obs = next_obs
    
    # Print episode summary
    print("\nEpisode finished:")
    print(f"Steps: {steps}/{max_steps}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Done: {done}, Truncated: {truncated}")
    
    if "is_success" in info:
        print(f"Success: {info['is_success']}")
    if "captures" in info:
        print(f"Captures: {info['captures']}")
    
    return total_reward, steps, info

def test_environment(action_type, use_predator=True, episodes=3, render=True):
    """Test the environment with specified action type and settings."""
    # Create environment
    env = DualEvadeEnv(
        world_name="21_05",
        use_lppos=True,
        use_predator=use_predator,
        max_step=300,
        reward_function=custom_reward,
        time_step=0.1,  # Faster time step for debugging
        render=render,
        real_time=render,
        end_on_pov_goal=True,
        use_other=False,  # Single agent mode for debugging
        action_type=action_type
    )
    
    # Print environment info
    print("\n" + "="*50)
    print(f"Testing environment with action_type: {action_type.name}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("="*50 + "\n")
    
    # Run multiple episodes
    episode_rewards = []
    episode_success = 0
    
    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes}")
        reward, steps, info = run_episode(env, verbose=(episode==0))
        episode_rewards.append(reward)
        
        if "is_success" in info and info["is_success"]:
            episode_success += 1
    
    # Print summary
    print("\nTest Results:")
    print(f"Average reward: {np.mean(episode_rewards):.4f}")
    print(f"Success rate: {episode_success/episodes:.2f}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug DualEvadeEnv with different action spaces")
    parser.add_argument("--action", choices=["discrete", "continuous"], default="continuous", 
                        help="Action space type to test")
    parser.add_argument("--no-predator", action="store_true", help="Disable predator")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    # Set action type based on argument
    if args.action == "discrete":
        action_type = DualEvadeEnv.ActionType.DISCRETE
    else:
        action_type = DualEvadeEnv.ActionType.CONTINUOUS
    
    # Run test
    test_environment(
        action_type=action_type,
        use_predator=not args.no_predator,
        episodes=args.episodes,
        render=not args.no_render
    ) 