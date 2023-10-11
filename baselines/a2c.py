from algorithm import TrainableAlgorithm
from stable_baselines3.a2c import A2C as StableA2C

class A2C(TrainableAlgorithm, StableA2C):
  """A Trainable extension to A2C 
  using Hyperparameter settings from https://arxiv.org/pdf/1711.09883.pdf"""
  # 
  def __init__(self, ent_schedule=lambda step: max((1 - (step/5e+5)), 0.01) * 0.1,
    learning_rate=lambda remaining: remaining * 5e-4, # Decay to 0 in 1e6 steps
    vf_coef=0.25, ent_coef=0.1, max_grad_norm=40, rms_prop_eps=0.1, **kwargs):
    self.ent_schedule = ent_schedule
    super(A2C, self).__init__(
      learning_rate=learning_rate, vf_coef=vf_coef, ent_coef=ent_coef,
      max_grad_norm=max_grad_norm, rms_prop_eps=rms_prop_eps, **kwargs
    )

  def train(self) -> None:
    if self.ent_schedule: 
      self.ent_coef = self.ent_schedule(self.num_timesteps)
      self.logger.record("train/env_coef", self.ent_coef)
    self.logger.record("rewards/environment", self.rollout_buffer.rewards.copy()) 
    super(A2C, self).train()
