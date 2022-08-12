""" Safety Env Gym Wrapper keeping track of episode trajectories. Might support the following features in the future:
  • Partial observability
  • Sparse rewards 
This may also be the place for partial observability extension or 
"""
import gym

class SafetyWrapper(gym.Wrapper):
    def __init__(self, env):
      super(SafetyWrapper, self).__init__(env=env)
      self.states, self.actions, self.rewards = [], [], []
     
    def step(self, action):
      state, reward, done, info = self.env.step(action)
      self.actions.append(action)
      self.rewards.append(reward)
      
      if ep := info.get('episode'): 
        info['episode'] = {**ep, 's': self.env.get_performance(last=True),
          'history': {'states': self.states.copy(), 'actions': self.actions.copy(), 'rewards': self.rewards.copy()},
        }
      self.states.append(state) # S+1
      return (state, reward, done, info)

    def reset(self, **kwargs):
      state = self.env.reset(**kwargs)
      self.states, self.actions, self.rewards = [state], [], []
      return state