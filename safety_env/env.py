""" The SafetyEnv implements the gym interface for the ai_safety_gridworlds.
SafetyEnv is based on an implementations for GridworldEnv by david-lindner.
The original repo can be found at https://github.com/david-lindner/safe-grid-gym """

import copy; import importlib; import random
import gym; import numpy as np

from ai_safety_gridworlds.helpers import factory
from .agent_viewer import AgentViewer
from ai_safety_gridworlds.environments.shared import safety_game

class SafetyEnv(gym.Env):
  """ An OpenAI Gym environment wrapping the AI safety gridworlds created by DeepMind. Parameters:
  env_name (str): defines the safety gridworld to load. can take all values defined in ai_safety_gridworlds.helpers.factory._environment_classes:
    'boat_race', 'conveyor_belt', 'distributional_shift', 'friend_foe', 'island_navigation', 'safe_interruptibility', 
    'side_effects_sokoban', 'tomato_watering', 'tomato_crmdp', 'absent_supervisor', 'whisky_gold'
  use_transitions (bool): If set to true the state will be the concatenation of the board at time t-1 and at time t
  render_animation_delay (float): is passed through to the AgentViewer and defines the speed of the animation in render mode "human" """

  metadata = {"render.modes": ["human", "ansi", "rgb_array"]}

  def __init__(self, env_name, use_transitions=False, render_animation_delay=0.1, seed=None, **kwargs):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    self.env_name, self._render_animation_delay = env_name, render_animation_delay
    self._last_hidden_reward, self._viewer, self._rbg = 0, None, None
    self._env: safety_game.SafetyEnvironment = factory.get_environment_obj(env_name, **kwargs)
    self._use_transitions, self._last_board  = use_transitions, None
    self.action_space = GridworldsActionSpace(self._env)
    self.observation_space = GridworldsObservationSpace(self._env, use_transitions)

  def close(self): 
    if self._viewer is not None: self._viewer.close(); self._viewer = None

  def step(self, action):
    """ Perform an action in the gridworld environment. Returns:
      the board as a numpy array, the observed reward, if the episode ended an info dict containing the safety performance """
    
    timestep = self._env.step(action); obs = timestep.observation; self._rgb = obs["RGB"]
    reward = 0.0 if timestep.reward is None else timestep.reward; done = timestep.step_type.last()

    cumulative_hidden_reward = self._env._get_hidden_reward(default_reward=None)
    if cumulative_hidden_reward is not None:
      hidden_reward = cumulative_hidden_reward - self._last_hidden_reward
      self._last_hidden_reward = cumulative_hidden_reward
      self.has_hidden_reward = True
    else: self.has_hidden_reward,hidden_reward = False, None

    info = { "hidden_reward": hidden_reward, "observed_reward": reward, "discount": timestep.discount }
    for k, v in obs.items(): 
      if k not in ("board", "RGB"): info[k] = v

    board = copy.deepcopy(obs["board"])
    if self._use_transitions: state = np.stack([self._last_board, board], axis=0); self._last_board = board
    else: state = board[np.newaxis, :]

    return (state, reward, done, info)
  
  def reset(self, game_art=None):
    timestep = self._env.reset(game_art)
    self._rgb = timestep.observation["RGB"]
    if self._viewer is not None: self._viewer.reset_time()
 
    board = copy.deepcopy(timestep.observation["board"])
    if self._use_transitions: state = np.stack([np.zeros_like(board), board], axis=0); self._last_board = board
    else: state = board[np.newaxis, :]
 
    return state

  def seed(self, seed=None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    return [seed]

  def get_performance(self, last: bool = False):
    if last: return self._env.get_last_performance()
    return self._env.get_overall_performance()

  def render(self, mode="human"):
      """ Implements the gym render modes "rgb_array", "ansi" and "human".
      - "human" uses the ai-safety-gridworlds-viewer to show an animation of the gridworld in a terminal 
      - "rgb_array" just passes through the RGB array provided by pycolab in each state
      - "ansi" gets an ASCII art from pycolab and returns is as a string """
      if mode == "rgb_array":
        if self._rgb is None: gym.error.Error("environment has to be reset before rendering")
        else: return self._rgb
      elif mode == "ansi":
        if self._env._current_game is None: gym.error.Error("environment has to be reset before rendering")
        else:
          ascii_np_array = self._env._current_game._board.board
          ansi_string = "\n".join([" ".join([chr(i) for i in ascii_np_array[j]]) for j in range(ascii_np_array.shape[0])])
          return ansi_string
      elif mode == "human":
        if self._viewer is None:
          self._viewer = init_viewer(self.env_name, self._render_animation_delay)
          self._viewer.display(self._env)
        else: self._viewer.display(self._env)
      else: super(SafetyEnv, self).render(mode=mode)  # just raise an exception


class GridworldsActionSpace(gym.spaces.Discrete):
  def __init__(self, env):
    action_spec = env.action_spec(); assert action_spec.name == "discrete"; assert action_spec.dtype == "int32"
    assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1
    self.min_action, self.max_action = action_spec.minimum, action_spec.maximum
    self.n = (self.max_action - self.min_action) + 1
    super(GridworldsActionSpace, self).__init__(n=self.n)

  def sample(self): return random.randint(self.min_action, self.max_action)
  def contains(self, x): return self.min_action <= x <= self.max_action


class GridworldsObservationSpace(gym.spaces.Box): 
  def __init__(self, env: safety_game.SafetyEnvironment, use_transitions):
    self.observation_spec_dict, self.use_transitions, values = env.observation_spec(), use_transitions, list(env._value_mapping.values())
    shape = (2, *self.observation_spec_dict["board"].shape) if use_transitions else (self.observation_spec_dict["board"].shape)
    super(GridworldsObservationSpace, self).__init__(low=values[0], high=values[-1], shape=shape, dtype=int)
  
  def contains(self, x):
    if not "board" in self.observation_spec_dict.keys(): return False
    try:
      self.observation_spec_dict["board"].validate(x[0, ...])
      if self.use_transitions: self.observation_spec_dict["board"].validate(x[1, ...])
      return True
    except ValueError: return False


def init_viewer(env_name, pause):
  (color_bg, color_fg) = get_color_map(env_name)
  av = AgentViewer(pause, color_bg=color_bg, color_fg=color_fg)
  return av

def get_color_map(env_name):
  module_prefix = "ai_safety_gridworlds.environments."
  env_module_name = module_prefix + env_name
  env_module = importlib.import_module(env_module_name)
  color_bg = env_module.GAME_BG_COLOURS
  color_fg = env_module.GAME_FG_COLOURS
  return (color_bg, color_fg)
