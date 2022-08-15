import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from . import DirectBuffer

class DirectCallback(BaseCallback):
  """ Callback for:
  • Calculating the DIRECT reward for every step based on the observation, action and environmental reward
  • when an episde is completed : updating the DIRECT Buffer & writing Accumulatied DIRECT returns to info
  • Keep a copy of the environmental rewards for logging, keep track of the acumulated DIRECT return  """

  def _on_training_start(self) -> None:
    self.model.rollout_buffer.real_rewards = np.zeros_like(self.model.rollout_buffer.rewards)
    self.disc_return = [0 for _ in range(self.model.n_envs)]
      
  def _on_step(self) -> bool: 
    l = lambda name: self.locals.get(name)

    # Get Observations actions and environment rewards, save real_reward for training discriminator
    observations, actions, real_rewards = self.model._last_obs, l('actions'), l('rewards')
    self.model.rollout_buffer.real_rewards[self.model.rollout_buffer.pos] = np.array(real_rewards).copy()

    # Calculate Discriminative Rewards for updating the policy, add to return 
    data = DirectBuffer.prepare(self.model.buffer, observations, actions, real_rewards, len=self.model.n_envs)
    disc_rewards = self.model.discriminator.reward(data).flatten(); self.disc_return += disc_rewards
    rewards = (self.model.chi * disc_rewards) + ((1-self.model.chi) * real_rewards)

    for i, (info, reward) in enumerate(zip(l('infos'), rewards)):
      self.locals['rewards'][i] = reward # Update rewards with direct reward (previous: self.action_probs as Bias)
      if ep := info.get('episode'): # Extend DIRECT Buffer \w Episde experience & write disc_return to episode info 
        self.model.buffer.extend(ep); ep['d']=self.disc_return[i]; self.disc_return[i] = 0
    return True
