from stable_baselines3.common.callbacks import CallbackList
from typing import Any, Dict, List, Tuple

from algorithm import TrainableAlgorithm
from . import DirectBuffer, Discriminator, DirectCallback

class DIRECT(TrainableAlgorithm):
  """ Discriminative Reward Co-Training. Extending the PPO Implementation by stable baselines 3
  :param env: The environment to learn from (if registered in Gym, can be str)
  :param kappa: (int) k-Best Trajectories to be stored in the reward buffer
  :param omega: (float) The frequency to perform updates to the discriminator. 
      [1/1 results in concurrent training of policy and discriminator]
      [1/2 results in 10 (n_updates) policy and 5 discriminator updates per rollout]
  :param chi: (float): The mixture parameter determining the mixture of real and discriminative reward
      [1 => purely discriminateive reward, 0 => no discriminateive reward, just real reward (pure PPO)]
  :param disc_kwargs: (dict) parameters to be passed the discriminator on creation
      see Discriminator class for full description
  :param **kwargs: further parameters will be passed to TrainableAlgorithm, then PPO"""

  def __init__(self, chi:float=None, kappa:int=None, omega:float=None, disc_kwargs:Dict[str,Any]={}, _init_setup_model=True,  **kwargs):
    self.buffer = None; self.discriminator, self.disc_kwargs = None, disc_kwargs  
    super().__init__(_init_setup_model=False, **kwargs); sparse = self.env.get_attr('sparse')[0]    
    self.chi = chi if chi is not None else 1.0 if sparse else 0.5; assert self.chi <= 1.0
    self.kappa = kappa if kappa is not None else 2048; assert self.kappa > 0
    self.omega = omega if omega is not None else 0.5 if sparse else 2.0; assert 0 < self.omega < 10
    self._suffix = f"{self.chi}_{self.omega}_{self.kappa}" # if args['algorithm'] == "DIRECT" else 'baseline'
    if _init_setup_model: self._setup_model()


  def _setup_model(self) -> None: 
    super(DIRECT, self)._setup_model(); self._naming.update({'d': 'direct-100'}) 
    self.heatmap_iterations.update( {'direct': (lambda _, s, a, r: self.discriminator.reward(
      DirectBuffer.prepare(self.buffer, [s], [a], [r()])
    ).flatten()[0], (None, None))})
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
    