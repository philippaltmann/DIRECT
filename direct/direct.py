from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import CallbackList
from typing import Any, Dict, List, Tuple

from util import TrainableAlgorithm
from . import DirectBuffer, Discriminator, DirectCallback

class DIRECT(TrainableAlgorithm, PPO):
  """ Discriminative Reward Co-Training. Extending the PPO Implementation by stable baselines 3
  :param env: The environment to learn from (if registered in Gym, can be str)
  :param kappa: (int) k-Best Trajectories to be stored in the reward buffer
  :param omega: (float) The frequency to perform updates to the discriminator. 
      [1/1 results in concurrent training of policy and discriminator]
  :param chi: (float): The mixture parameter determining the mixture of real and discriminative reward
      [1 => purely discriminateive reward, 0 => no discriminateive reward, just real reward (pure PPO)]
  :param disc_kwargs: (dict) parameters to be passed the discriminator on creation
      see Discriminator class for full description
  :param **kwargs: further parameters will be passed to TrainableAlgorithm, then PPO"""

  def __init__(self, chi:float=1., kappa:int=512, omega:float=1/1, disc_kwargs:Dict[str,Any]={}, **kwargs):
    self.buffer = None
    self.chi = chi; assert chi <= 1.0
    self.kappa = kappa; assert kappa > 0
    self.omega = omega; assert 0 < omega < 10
    self.discriminator, self.disc_kwargs = None, disc_kwargs  
    super().__init__(**kwargs)

  def _setup_model(self) -> None: 
    super(DIRECT, self)._setup_model(); self._naming.update({'d': 'direct-100'}) 
    self.disc_kwargs.setdefault('batch_size', self.batch_size) 
    self.buffer = DirectBuffer(buffer_size=self.kappa, parent=self)
    self.discriminator = Discriminator(chi=self.chi, parent=self, **self.disc_kwargs).to(self.device)

  def _excluded_save_params(self) -> List[str]:  return super(DIRECT, self)._excluded_save_params() + ['buffer'] 

  def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
    torch, vars = super(DIRECT, self)._get_torch_save_params()
    torch.extend(["discriminator", "discriminator.optimizer"])
    return torch, vars

  def learn(self, reset_num_timesteps: bool = True, **kwargs) -> "TrainableAlgorithm":
    kwargs['callback'] = CallbackList([DirectCallback(), kwargs.pop('callback')]) if 'callback' in kwargs else DirectCallback()
    return super(DIRECT, self).learn(reset_num_timesteps=reset_num_timesteps, **kwargs)

  def train(self) -> None:
    # Train Discriminator
    self.discriminator.train(buffer=self.buffer, rollout=self.rollout_buffer)

    # Write metrics
    self.logger.record("discriminator", self.discriminator.metrics())
    self.logger.record("buffer", self.buffer.metrics())
    self.logger.record("rewards/environment", self.rollout_buffer.real_rewards.copy()) 

    # Train PPO
    super(DIRECT, self).train()
    