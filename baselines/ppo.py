from algorithm import TrainableAlgorithm
from stable_baselines3.ppo import PPO as StablePPO

class PPO(TrainableAlgorithm, StablePPO):
  """A Trainable extension to PPO"""
  def train(self) -> None:
    self.logger.record("rewards/environment", self.rollout_buffer.rewards.copy()) 
    super(PPO, self).train()
