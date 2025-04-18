from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import numpy as np

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
        # both agents observe the same initial obs
        self.agents = list(self.possible_agents)
        observations = {agent: obs0.copy() for agent in self.agents}
        return observations

    def step(self, actions):
        # actions: dict mapping agent to its chosen action
        action_prey1 = actions.get("prey_1")
        action_prey2 = actions.get("prey_2")
        # inject other agent action via inner_env.set_other_policy
        self.inner_env.set_other_policy(lambda obs: action_prey2)
        # step inner env with primary agent's action
        obs, reward, done, truncated, info = self.inner_env.step(action_prey1)

        # build parallel return
        observations = {agent: obs.copy() for agent in self.agents}
        rewards = {agent: reward for agent in self.agents}
        # done flags per agent
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        infos = {agent: info for agent in self.agents}
        return observations, rewards, dones, infos

    def render(self):
        # delegate to inner env's render via model
        return None

    def close(self):
        self.inner_env.close()

# wrapper function for default usage
# No OrderEnforcingWrapper needed for ParallelEnv

def env(**kwargs):
    return MultiDualEvadeEnv(**kwargs)

