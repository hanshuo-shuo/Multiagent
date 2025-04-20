from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import numpy as np
from env import DualEvadeObservation

class MultiDualEvadeEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for DualEvadeEnv.
    Wraps existing DualEvadeEnv to expose both prey agents as independent agents.
    """
    metadata = {"render_modes": ["human"], "name": "multi_dual_evade_parallel"}

    def __init__(self, **dual_evade_kwargs):
        super().__init__()
        # instantiate the original single-agent env
        from env import DualEvadeEnv
        self.inner_env = DualEvadeEnv(**dual_evade_kwargs)
        
        # Create another observation object for the second agent
        self.prey2_observation = DualEvadeObservation()
        
        # Store the reward function
        self.reward_function = self.inner_env.reward_function

        # define agent names
        self.agents = ["prey_1", "prey_2"]
        self.possible_agents = list(self.agents)

        # use same observation and action space for both
        obs_space = self.inner_env.observation_space
        act_space = self.inner_env.action_space
        self.observation_spaces = {agent: obs_space for agent in self.agents}
        self.action_spaces = {agent: act_space for agent in self.agents}

    def reset(self, seed=None, options=None):
        # reset inner env
        obs0, _ = self.inner_env.reset(options=options, seed=seed)
        # Get initial observations for both agents
        self.agents = list(self.possible_agents)
        
        # For prey_1, use the observation returned by the inner env
        prey1_obs = obs0.copy()
        
        # For prey_2, generate a swapped observation
        prey2_obs = self.__get_prey2_observation()
        
        observations = {
            "prey_1": prey1_obs,
            "prey_2": prey2_obs
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # actions: dict mapping agent to its chosen action
        action_prey1 = actions.get("prey_1")
        action_prey2 = actions.get("prey_2")
        # inject other agent action via inner_env.set_other_policy
        self.inner_env.set_other_policy(lambda obs: action_prey2)
        # step inner env with primary agent's action
        obs, reward_prey1, terminated, truncated, info = self.inner_env.step(action_prey1)

        # Get prey_1's observation
        prey1_obs = obs.copy()
        
        # Get prey_2's observation
        prey2_obs = self.__get_prey2_observation()
        
        # Calculate reward for prey_2 using its observation
        reward_prey2 = self.reward_function(prey2_obs)
        
        # build parallel return
        observations = {
            "prey_1": prey1_obs,
            "prey_2": prey2_obs
        }
        rewards = {
            "prey_1": reward_prey1,
            "prey_2": reward_prey2
        }
        # terminated and truncated flags per agent
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: info.copy() if info else {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos
        
    def __get_prey2_observation(self):
        """Generate the observation from prey_2's perspective"""
        # Get the model from inner_env
        model = self.inner_env.model
        
        # Update observation for prey_2 perspective
        self.inner_env.__update_observation__(
            observation=self.prey2_observation,
            prey=model.prey_2,
            prey_data=model.prey_data_2,
            other=model.prey_1
        )
        
        return self.prey2_observation.copy()

    def render(self):
        # delegate to inner env's render via model
        return None

    def close(self):
        self.inner_env.close()

# wrapper function for default usage
# No OrderEnforcingWrapper needed for ParallelEnv

def env(**kwargs):
    return MultiDualEvadeEnv(**kwargs)

