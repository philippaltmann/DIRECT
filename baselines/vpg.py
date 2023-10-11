from algorithm import TrainableAlgorithm
from stable_baselines3.a2c import A2C

class VPG(TrainableAlgorithm, A2C):
  """A Trainable extension to A2C"""
  def __init__(self, **kwargs): 
    super(VPG, self).__init__(
      learning_rate=3e-4, n_steps=1000, gae_lambda=1.0, vf_coef=0.0, 
      use_rms_prop=False, normalize_advantage=True, **kwargs) 

  def train(self) -> None:
    self.logger.record("rewards/environment", self.rollout_buffer.rewards.copy()) 
    super(VPG, self).train()
