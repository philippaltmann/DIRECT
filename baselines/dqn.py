import torch as th; import numpy as np
from algorithm import TrainableAlgorithm
from stable_baselines3.dqn import DQN as StableDQN

class DQN(TrainableAlgorithm, StableDQN):
  """A Trainable extension to DQN"""
  def __init__(self, **kwargs): 
    super(DQN, self).__init__(**kwargs); 
    self.n_steps = self.train_freq if type(self.train_freq) == int else self.train_freq[0]

  def _setup_model(self) -> None:
    super(DQN, self)._setup_model()
    def ga(s): a = self.policy.q_net.forward(th.tensor([s])); return a/a.sum()
    # self.get_actions = lambda s: self.policy.q_net.forward(th.tensor([s]))#.cpu().detach().numpy()
    self.get_actions = lambda s: ga(s)


  def train(self, **kwargs) -> None:
    self.logger.record("rewards/environment", self.replay_buffer.rewards.copy()) 
    super(DQN, self).train(**kwargs); self.prev_steps = self.num_timesteps
