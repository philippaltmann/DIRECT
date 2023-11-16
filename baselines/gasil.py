import numpy as np
from direct import DIRECT as BaseDIRECT
from stable_baselines3.ppo import PPO

class GASIL(BaseDIRECT, PPO):
  """A Trainable extension to PPO using Generative Adversarial Self-Imitation Leanrning 
  (https://openreview.net/pdf?id=HJeABnCqKQ)
  buffersize: 1000  => kappa=1000
  updates:    20    => omega=2 
  alpha:      0.1   => chi = 0.1 
  mixture = lambda r,d,c: r-c*np.log(d) """
  def __init__(self, **kwargs):
    super().__init__(kappa=1000, omega=2, chi=0.1,
                      mixture = lambda r,d,c: r-c * np.log(d+np.finfo(np.float64).resolution), 
                      **kwargs)
    self._suffix = '' #Overwrite DIRECT Suffix containing HP
