import numpy as np
import time
import argparse
from env import DualEvadeEnv, DualEvadeObservation
from pz_env import MultiDualEvadeEnv

def custom_reward(obs: DualEvadeObservation) -> float:
    """
    Simple reward function for debugging:
    - Penalty for getting puffed
    - Big reward for reaching goal
    """
    if obs.puffed > 0:
        return -1
    else:
        # Episode is finished
        if obs.prey_goal_distance < 0.1:  # Reached goal
            return 1
        return 0.0  # Terminated but didn't reach goal

def print_observation(agent_id, obs):
    """Print observation details for a particular agent"""
    print(f"Observation for {agent_id}:")
    if isinstance(obs, np.ndarray):
        print(f"  Type: numpy.ndarray, Shape: {obs.shape}")
        # Map array indices to field names
        fields = DualEvadeObservation.fields
        for i, field in enumerate(fields):
            print(f"    {field}: {obs[i]:.4f}")
    else:
        print(f"  Type: {type(obs)}")
        for field in DualEvadeObservation.fields:
            value = getattr(obs, field)
            print(f"    {field}: {value:.4f}")

def run_episode(env, max_steps=300, verbose=True, sleep_time=0.01):
    """Run a single episode with the given multi-agent environment."""
    observations, _ = env.reset()
    
    # Print initial observations for each agent
    if verbose:
        print("Initial observations:")
        for agent_id, obs in observations.items():
            print_observation(agent_id, obs)
    
    done = False
    total_rewards = {agent_id: 0.0 for agent_id in env.agents}
    steps = 0
    
    # Run episode
    while not done and steps < max_steps:
        # Sample actions for each agent
        actions = {}
        action_descs = {}
        
        for agent_id in env.agents:
            action_space = env.action_spaces[agent_id]
            
            if hasattr(action_space, 'n'):  # Check for discrete space
                # Discrete action space
                action = np.random.randint(0, action_space.n)
                action_descs[agent_id] = f"Discrete: {action}"
            else:
                # Continuous action space
                action = action_space.sample()
                action_descs[agent_id] = f"Continuous: ({action[0]:.4f}, {action[1]:.4f})"
            
            actions[agent_id] = action
        
        # Step environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check if episode is done (all agents terminated or truncated)
        done = all(terminations.values()) or all(truncations.values())
        
        # Update stats
        for agent_id in env.agents:
            total_rewards[agent_id] += rewards[agent_id]
        steps += 1
        
        # Log step information
        if verbose and steps % 10 == 0:
            print(f"\nStep {steps}:")
            for agent_id in env.agents:
                print(f"  {agent_id}: Action {action_descs[agent_id]}")
                print(f"    Reward: {rewards[agent_id]:.4f}, Total: {total_rewards[agent_id]:.4f}")
                print(f"    Position: ({next_observations[agent_id][0]:.4f}, {next_observations[agent_id][1]:.4f})")
                print(f"    Goal distance: {next_observations[agent_id][9]:.4f}")
        
        # Small delay for visualization
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Update observations
        observations = next_observations
    
    # Print episode summary
    print("\nEpisode finished:")
    print(f"Steps: {steps}/{max_steps}")
    print("Total rewards:")
    for agent_id in env.agents:
        print(f"  {agent_id}: {total_rewards[agent_id]:.4f}")
    
    # Print success information
    if done:
        print("Success status:")
        for agent_id in env.agents:
            if agent_id in infos and "is_success" in infos[agent_id]:
                print(f"  {agent_id}: {infos[agent_id]['is_success']}")
            if agent_id in infos and "captures" in infos[agent_id]:
                print(f"  {agent_id} captures: {infos[agent_id]['captures']}")
    
    return total_rewards, steps, infos

def test_multi_environment(action_type, use_predator=True, episodes=3, render=True):
    """Test the multi-agent environment with specified action type and settings."""
    # Create environment
    env = MultiDualEvadeEnv(
        world_name="21_05",
        use_lppos=True,
        use_predator=use_predator,
        max_step=300,
        reward_function=custom_reward,
        time_step=0.1,  # Faster time step for debugging
        render=render,
        real_time=render,
        end_on_pov_goal=False,  # Both agents need to reach goal
        use_other=True,
        action_type=action_type
    )
    
    # Print environment info
    print("\n" + "="*50)
    print(f"Testing multi-agent environment with action_type: {action_type.name}")
    print(f"Agents: {env.agents}")
    for agent_id in env.agents:
        print(f"Observation space for {agent_id}: {env.observation_spaces[agent_id]}")
        print(f"Action space for {agent_id}: {env.action_spaces[agent_id]}")
    print("="*50 + "\n")
    
    # Run multiple episodes
    all_rewards = {agent_id: [] for agent_id in env.agents}
    success_counts = {agent_id: 0 for agent_id in env.agents}
    
    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes}")
        rewards, steps, infos = run_episode(env, verbose=(episode==0))
        
        # Record rewards and success
        for agent_id in env.agents:
            all_rewards[agent_id].append(rewards[agent_id])
            if agent_id in infos and "is_success" in infos[agent_id] and infos[agent_id]["is_success"]:
                success_counts[agent_id] += 1
    
    # Print summary
    print("\nTest Results:")
    for agent_id in env.agents:
        avg_reward = np.mean(all_rewards[agent_id])
        success_rate = success_counts[agent_id] / episodes
        print(f"{agent_id} - Avg reward: {avg_reward:.4f}, Success rate: {success_rate:.2f}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug MultiDualEvadeEnv with different action spaces")
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

    print(f"Testing with action type: {action_type.name}")
    
    # Run test
    test_multi_environment(
        action_type=action_type,
        use_predator=not args.no_predator,
        episodes=args.episodes,
        render=not args.no_render
    ) 