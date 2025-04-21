import random
import typing
import cellworld_game as cwgame
import numpy as np
import math
import enum

from cellworld_game import AgentState
from gymnasium import Env
from gymnasium import spaces

import typing
import numpy as np
from enum import Enum
from gymnasium import Env


class Observation(np.ndarray):
    fields = []  # list of field names in the observation

    def __init__(self):
        super().__init__()
        for index, field in enumerate(self.__class__.fields):
            self._create_property(index=index,
                                  field=field)
        self.field_enum = Enum("fields", {field: index for index, field in enumerate(self.__class__.fields)})

    def __new__(cls):
        # Create a new array of zeros with the given shape and dtype
        shape = (len(cls.fields),)
        dtype = np.float32
        buffer = None
        offset = 0
        strides = None
        order = None
        obj = super(Observation, cls).__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.fill(0)
        return obj

    def _create_property(self,
                         index: int,
                         field: str):
        def getter(self):
            return self[index]

        def setter(self, value):
            self[index] = value

        setattr(self.__class__, field, property(getter, setter))

    def __setitem__(self, field: typing.Union[Enum, int], value):
        if isinstance(field, Enum):
            np.ndarray.__setitem__(self, field.value, value)
        else:
            np.ndarray.__setitem__(self, field, value)

    def __getitem__(self, field: typing.Union[Enum, int]) -> np.ndarray:
        if isinstance(field, Enum):
            return np.ndarray.__getitem__(self, field.value)
        else:
            return np.ndarray.__getitem__(self, field)




class Environment(Env):
    def __init__(self):
        self.event_handlers: typing.Dict[str, typing.List[typing.Callable]] = {"reset": [],
                                                                               "step": []}

    def __handle_event__(self, event_name: str, *args):
        for handler in self.event_handlers[event_name]:
            handler(*args)

    def add_event_handler(self, event_name: str, handler: typing.Callable):
        if event_name not in self.event_handlers:
            raise "Event handler not registered"
        self.event_handlers[event_name].append(handler)

    def reset(self,
              options: typing.Optional[dict] = None,
              seed=None):
        self.__handle_event__("reset", options, seed)

    def step(self, action: int):
        self.__handle_event__("step", action)




class DualEvadeObservation(Observation):
    fields = ["self_x",
              "self_y",
              "self_direction",
              "other_x",
              "other_y",
              "other_direction",
              "predator_x",
              "predator_y",
              "predator_direction",
              "prey_goal_distance",
            #   "predator_prey_distance",
              "puffed",
              "puff_cooled_down",
              "finished"]


class DualEvadeEnv(Environment):

    class ActionType(enum.Enum):
        DISCRETE = 0
        CONTINUOUS = 1

    def __init__(self,
                 world_name: str,
                 use_lppos: bool,
                 use_predator: bool,
                 max_step: int = 300,
                 reward_function: typing.Callable[[DualEvadeObservation], float] = lambda x: 0,
                 time_step: float = .25,
                 render: bool = False,
                 real_time: bool = False,
                 end_on_pov_goal: bool = True,
                 use_other: bool = True,
                 action_type: ActionType = ActionType.DISCRETE):
        self.max_step = max_step
        self.reward_function = reward_function
        self.time_step = time_step
        self.loader = cwgame.CellWorldLoader(world_name=world_name)
        self.use_other = use_other
        self.action_type = action_type
        
        if self.use_other:
            if self.action_type == DualEvadeEnv.ActionType.DISCRETE:
                self.other_policy = lambda x: random.randint(0, len(self.action_list) - 1)
            else:
                self.other_policy = lambda x: np.random.random(2)
                
            self.model = cwgame.DualEvade(world_name=world_name,
                                          real_time=real_time,
                                          render=render,
                                          use_predator=use_predator)
            self.prey = self.model.prey_1
            self.other = self.model.prey_2
            self.prey_data = self.model.prey_data_1
            self.other_data = self.model.prey_data_2
            self.end_on_pov_goal = end_on_pov_goal
        else:
            if self.action_type == DualEvadeEnv.ActionType.DISCRETE:
                self.other_policy = None
            else:
                self.other_policy = None
                
            self.model = cwgame.BotEvade(world_name=world_name,
                                         real_time=real_time,
                                         render=render,
                                         use_predator=use_predator)
            self.prey = self.model.prey
            self.other = None
            self.prey_data = self.model.prey_data
            self.other_data = None
            self.end_on_pov_goal = True
            
        self.other_observation = DualEvadeObservation()
        self.observation = DualEvadeObservation()
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self.observation),), dtype=np.float32)
        
        # Setup action space based on type
        if use_lppos:
            self.action_list = self.loader.tlppo_action_list
        else:
            self.action_list = self.loader.full_action_list

        if self.action_type == DualEvadeEnv.ActionType.DISCRETE:
            self.action_space = spaces.Discrete(len(self.action_list))
        else:
            self.action_space = spaces.Box(0.0, 1.0, (2,), dtype=np.float32)

        self.prey_trajectory_length = 0
        self.predator_trajectory_length = 0
        self.episode_reward = 0
        self.step_count = 0
        Environment.__init__(self)

    def set_other_policy(self, other_policy):
        self.other_policy = other_policy

    def __update_observation__(self, observation: DualEvadeObservation, prey, prey_data, other):
        observation.self_x = float(prey.state.location[0])
        observation.self_y = float(prey.state.location[1])
        observation.self_direction = float(math.radians(prey.state.direction))

        if self.use_other:
            if self.model.mouse_visible:
                observation.other_x = float(other.state.location[0])
                observation.other_y = float(other.state.location[1])
                observation.other_direction = float(math.radians(other.state.direction))
            else:
                observation.other_x = 0.0
                observation.other_y = 0.0
                observation.other_direction = 0.0
        else:
            observation.other_x = 0.0
            observation.other_y = 0.0
            observation.other_direction = 0.0

        if self.model.use_predator and prey_data.predator_visible:
            observation.predator_x = float(self.model.predator.state.location[0])
            observation.predator_y = float(self.model.predator.state.location[1])
            observation.predator_direction = float(math.radians(self.model.predator.state.direction))
        else:
            observation.predator_x = 0.0
            observation.predator_y = 0.0
            observation.predator_direction = 0.0

        observation.prey_goal_distance = float(prey_data.prey_goal_distance)
        # observation.predator_prey_distance = prey_data.predator_prey_distance

        observation.puffed = float(prey_data.puffed)
        observation.puff_cooled_down = float(self.model.puff_cool_down)
        observation.finished = float(not self.model.running)
        return observation

    def set_actions(self, action, other_action=None):
        if self.action_type == DualEvadeEnv.ActionType.DISCRETE:
            self.prey.set_destination(self.action_list[action])
            if other_action is not None:
                self.other.set_destination(self.action_list[other_action])
        else:
            self.prey.set_destination(tuple(action))
            if other_action is not None:
                self.other.set_destination(tuple(other_action))

    def __step__(self):
        self.step_count += 1
        truncated = (self.step_count >= self.max_step)

        obs = self.__update_observation__(observation=self.observation,
                                          prey=self.prey,
                                          prey_data=self.prey_data,
                                          other=self.other)
        reward = self.reward_function(obs)
        self.episode_reward += reward

        if self.prey_data.puffed:
            self.prey_data.puffed = False

        if self.prey_data.goal_achieved:
            if self.end_on_pov_goal or self.other_data is None or self.other_data.goal_achieved:
                self.model.stop()

        if not self.model.running or truncated:
            survived = 1 if not self.model.running and self.prey_data.puff_count == 0 else 0
            info = {"captures": self.prey_data.puff_count,
                    "reward": self.episode_reward,
                    "is_success": survived,
                    "survived": survived,
                    "agents": {}}
        else:
            info = {}
        return obs, reward, not self.model.running, truncated, info

    def replay_step(self, agents_state: typing.Dict[str, AgentState]):
        self.model.set_agents_state(agents_state=agents_state,
                                    delta_t=self.time_step)
        return self.__step__()

    def step(self, action):
        if self.use_other:
            other_obs = self.__update_observation__(observation=self.other_observation,
                                                    prey=self.other,
                                                    prey_data=self.other_data,
                                                    other=self.prey)
            other_action = self.other_policy(other_obs)
            self.set_actions(action=action,
                             other_action=other_action)
        else:
            self.set_actions(action=action)

        model_t = self.model.time + self.time_step
        while self.model.running and self.model.time < model_t:
            self.model.step()
        Environment.step(self, action=action)
        return self.__step__()

    def __reset__(self):
        self.episode_reward = 0
        self.step_count = 0
        return self.__update_observation__(observation=self.observation,
                                           prey=self.prey,
                                           prey_data=self.prey_data,
                                           other=self.other), {}

    def reset(self,
              options={},
              seed=None):
        self.model.reset()
        Environment.reset(self, options=options, seed=seed)
        return self.__reset__()

    def replay_reset(self, agents_state: typing.Dict[str, AgentState]):
        self.model.reset()
        self.model.set_agents_state(agents_state=agents_state)
        return self.__reset__()

    def close(self):
        self.model.close()
        Env.close(self=self)