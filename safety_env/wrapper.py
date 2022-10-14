""" Safety Env Gym Wrapper keeping track of episode trajectories. Might support the following features in the future:
  • Partial observability
  • Sparse / delayed rewards 
This may also be the place for partial observability extension or 
"""
import gym; import numpy as np
from ai_safety_gridworlds.environments.shared import safety_game
from .plotting import heatmap_2D

class SafetyWrapper(gym.Wrapper):
    def __init__(self, env, sparse=True):
      super(SafetyWrapper, self).__init__(env=env)
      self.states, self.actions, self.rewards, self.sparse = [], [], [], sparse
      self._history = lambda: {
        'states': np.array(self.states.copy()), 
        'actions': np.array(self.actions.copy()), 
        'rewards': np.array(self.rewards.copy())
      }
     
    def step(self, action):
      state, reward, done, info = self.env.step(action)
      self.actions.append(action); self.rewards.append(reward)
      
      if ep := info.get('episode'): 
        ep.update({**ep, 's': self.env.get_performance(last=True), 'history': self._history()})
      
      self.states.append(state) # S+1
      if self.sparse and not done: return (state, 0.0, done, info)
      return (state, reward, done, info)

    def reset(self, game_art=None):
      state = self.env.reset(game_art=game_art)
      self.states, self.actions, self.rewards = [state], [], []
      return state

    def iterate(self, function = lambda e,s,a,r: r(), agent='A', field=' ', fallback=None):
      """Iterate all possible actions in all env states, apply `funcion(env, state, action)`
      function: `f(env, state, action, reward()) => value` to be applied to all actions in all states 
        default: return envreward
      fields: Mapping for fields to replace, agent: Mapping of agent field to move in env
      fallback: result to put in fields where agent can't be moved (eg. walls)
      :returns: ENVxACTIONS shaped function results"""

      # Get env object, prepare mappings for env generation 
      safety_env = self.env.env.env; empty_board = self.reset()[0]
      value_mapping = safety_env._env._value_mapping; act = safety_game.Actions
      agent_value = value_mapping[agent]; field_value = value_mapping[field]
      actions = [act.UP, act.RIGHT, act.DOWN, act.LEFT]; fallback = [fallback] * len(actions)
      lookup = {v: k for k, v in value_mapping.items()}
      
      # Create empty board for iteration & function for reverting Observation to board 
      empty_board[empty_board == agent_value] = field_value
      board = lambda state: ["".join([lookup[cell] for cell in row]) for row in state]
      reward = lambda action: lambda: self.step(action)[1]
      def prepare(x,y): state = empty_board.copy(); state[y][x] = agent_value; return state
      process = lambda state: [function(self, self.reset(board(state)), action, reward(action)) for action in actions]
      
      return [ 
        [ process(prepare(x,y)) if cell == field_value else fallback for x, cell in enumerate(row) ] 
          for y, row in enumerate(empty_board)
      ]

    def heatmap(self, fn, args): return heatmap_2D(self.iterate(fn), *args)
      