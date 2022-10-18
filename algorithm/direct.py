from direct import DIRECT as BaseDIRECT
from stable_baselines3.ppo import PPO

class DIRECT(BaseDIRECT, PPO):
  """A Trainable extension to DIRECT using PPO Updates"""
