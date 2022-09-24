""" Safety Env Gym Wrapper keeping track of episode trajectories. Might support the following features in the future:
  • Partial observability
  • Sparse / delayed rewards 
This may also be the place for partial observability extension or 
"""
import gym; import numpy as np

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