"""
Discriminator Network Architecture, Loss, Update, Batchsampling
Inspired by: Generative Adversarial Imitation Learning https://arxiv.org/pdf/1606.03476.pdf
Adapted from https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/algorithms/adversarial/gail.py

Example Values: discriminator(sample)
    __________________________________________________________
    | forward | backward | logit | interpretation |  reward  |
    |–––––––––+––––––––––+–––––––+––––––––––––––––+––––––––––|
    |  - 100  | 0.0      |   0   | DIRECT buffer  | 100.0000 |
    |  -  10  | 0.00004  |   0   | DIRECT buffer  |  10.0000 |
    |  -   1  | 0.26894  |   0   | DIRECT buffer  |   1.3133 |
    |      0  | 0.5      |   0   | DIRECT buffer  |   0.6931 |
    |  +   1  | 0,73106  |   1   | current policy |   0.3132 |
    |  +  10  | 0.99995  |   1   | current policy |   4.5e-05|
    |  + 100  | 1.0      |   1   | current policy |   0.0000 |
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    ReLU       Sigmoid                             -logsigmoid

"""

import numpy as np; import torch as th
from typing import Dict, Any, List
from gym.spaces.utils import flatdim as _dim
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.base_class import BaseAlgorithm

from . import DirectBuffer, DirectBufferSamples 

class Discriminator(th.nn.Module): 
    """The discriminator to use for DIRECT."""

    def __init__(
        self, batch_size: int, chi: float, hidden_size: List[int]=[32,32], 
        optimizer_class: th.optim.Optimizer = th.optim.Adam, optim_kwargs: Dict[str, Any] = {},
        activation:th.nn.Module=th.nn.ReLU, parent: BaseAlgorithm = None, 
        use_obs:bool=True, use_act:bool=True, use_rew:bool=True, 
    ):
        """Construct discriminator network. Args:
          env: the used environment (used to calculate the network dimensions
            & normalize batches of training data)
          batch_size: the batch size for training the discriminator.
          chi: mixture of real and discriminative rewards 
          hidden_size: Hidden size of the discriminator network
          optimizer_class: the optimizer class to use for training the disc
          optim_kwargs: additional parameters to be passed the optimizer
            defaults to lower lr, as suggested for gans
          activation: activation to apply after hidden layers.
          use_obs, use_act, use_rew: should the current observation/action/reward 
            be included as an input to the MLP
          parent: ref to direct """
        super().__init__()
        self.activation = activation; self.batch_size = batch_size
        self.chi = chi; self.device = parent.device; self.parent = parent
        self.obs_dim = _dim(self.parent.env.unwrapped.observation_space) * use_obs
        self.act_dim = _dim(self.parent.env.unwrapped.action_space) * use_act
        self.rew_dim = 1 * use_rew; self.hidden_size = hidden_size
        self._losses, self._logits, self._labels = [], th.empty((0,1)), th.empty((0,1))

        layer = lambda i,o: [th.nn.Linear(i,o), self.activation()]
        units =  [self.obs_dim + self.act_dim + self.rew_dim] + self.hidden_size + [1]
        layers = [l for i, o in zip(units, units[1:]) for l in layer(i,o)][:-1]
        self.net = th.nn.Sequential(*layers)   # Lower lr as suggested for gans
        self.optimizer_class = optimizer_class; optim_kwargs.setdefault('lr', 2e-4)
        self.optimizer = self.optimizer_class(self.parameters(), **optim_kwargs)
        
        self.n_updates = int(self.parent.omega * self.parent.n_epochs); self.step = 0 
        assert self.n_updates > 0, "Discriminator update frequencies below 1/n_epoch are not supported"

    def forward(self, obs:th.Tensor, act:th.Tensor, rew:th.Tensor) -> th.Tensor:
        """Compute the discriminator's logits for a state-action-reward sample. Args:
            obs: observation at time t.
            act: action taken at time t.
            rew: reward received by the environmet for action at time t.
        Returns: preds: discriminative predictions for given activation (default: ReLU)
            high (>0) -> current policy (generator)
            low (<=0) -> imitation buffer (expert) """ 
        inputs = []
        if self.obs_dim: inputs.append(th.flatten(obs, 1))
        if self.act_dim: inputs.append(th.flatten(act, 1))
        if self.rew_dim: inputs.append(th.flatten(rew, 1))
        inputs_concat = th.cat(inputs, dim=1)
        return self.net(inputs_concat)

    def backward(self, obs:th.Tensor, act:th.Tensor, rew:th.Tensor, labels:th.Tensor) -> Dict[str, float]:
        """Perform a backward step, i.e. compute BCE-Loss, step optimizer, update parameters
        Apply binary_cross_entropy_with_logits in two steps (1.sigmoid, 2.binary_cross_entropy) 
        generates sigmoid-processed logits from batch: ]0;0.5]->pred_buffer ]0.5;1[->pred_policy
        Args: (parsed from the batch dict generated by `prepare_batch()`)
            obs: batch of observations,  act: batch of actions,  rew: batch of rewards 
            labels: integer labels, with 0->buffer, 1->policy (c.f. prediction legend)
        Returns: stats: a dict containing training stats."""        
        logits = th.sigmoid(self.forward(obs, act, rew)); self.optimizer.zero_grad(); 
        loss = th.nn.functional.binary_cross_entropy(logits, labels.float())
        loss.backward(); self.optimizer.step()
        return loss.cpu().detach().numpy(), logits, labels

    def reward(self, samples: DirectBufferSamples) -> np.ndarray:
        """Compute the discriminative rewards for a buffer of samples
        Normalization of DirectBufferSamples is handled in buffer.py
        Args: samples: Buffer of experience
        Returns: rewards: discriminative rewards, adapted for learning
            high  ->  imitation buffer     |     low   ->  current policy """
        obs = th.as_tensor(samples.observations, device=self.device)
        act = th.as_tensor(samples.actions, device=self.device)
        rew = th.as_tensor(samples.rewards, device=self.device)
        with th.no_grad(): reward = - th.nn.functional.logsigmoid(self.forward(obs=obs, act=act, rew=rew))
        assert reward.shape == rew.shape, "Reward should be of shape {}, but is {}!".format(rew.shape,reward.shape)
        return reward.cpu().numpy()
    
    def prepare_batch(self, policy_buffer: DirectBuffer, direct_buffer: DirectBuffer) -> Dict[str, th.Tensor]:
        """ Build and return training batch for the next discriminator update. Args:
            direct_buffer: Buffer to sample expert.
            policy_buffer: Buffer to sample policy.
        Returns: batch: A batch of policy and buffer training samples. """
        buffer_samples: DirectBufferSamples = direct_buffer.sample(self.batch_size)
        policy_samples: DirectBufferSamples = policy_buffer.sample(self.batch_size)
        assert (len(buffer_samples.observations) == len(policy_samples.observations) == self.batch_size), "Missmatching batch sizes!"

        # Concatenate rollouts, and label each row as expert or generator (_gen_is_one)
        obs = th.cat([buffer_samples.observations, policy_samples.observations])
        act = th.cat([buffer_samples.actions, policy_samples.actions])
        rew = th.cat([buffer_samples.rewards, policy_samples.rewards])
        labels = th.cat([ th.zeros_like(buffer_samples.rewards, device=self.device, dtype=int), 
                          th.ones_like(policy_samples.rewards, device=self.device, dtype=int), ])
        return { "obs": obs, "act": act, "rew": rew, "labels": labels}

    def train(self, buffer: DirectBuffer, rollout: RolloutBuffer):
        """ Perform {n_updates} discriminator update(s): Generate Batch, perform a backward/training step & log results. Args: 
          buffer: DirectBuffer to sample expert. rollout: RolloutBuffer to sample policy."""
        policy = DirectBuffer(buffer_size=rollout.buffer_size * rollout.n_envs , parent=self.parent).fill(
          rollout.observations, rollout.actions, rollout.real_rewards)
        for _ in range(self.n_updates):  
          for _ in range((policy.size() + buffer.size()) // (self.batch_size * 2)):
            batch = self.prepare_batch(policy_buffer=policy, direct_buffer=buffer)
            ls, lg, lb = self.backward(**batch); self._losses.append(ls)
            self._logits, self._labels = th.cat((self._logits,lg.cpu())), th.cat((self._labels,lb.cpu()))
          self.step += 1; 

    def metrics(self, reset=True):
      """Statistics for the  discriminator training averaged over all updates after reset:
        loss: binary crossentropy loss, acccuracy: portion of correct predictions, 
        entropy: over predicted label distribution (if this drops then disc is very good or has given up)
        policy / buffer: Portions of buffer and policy predictions """
      losses, logits, labels = self._losses.copy(), self._logits.clone(), self._labels.clone()
      if reset: self._losses, self._logits, self._labels = [], th.empty((0,1)), th.empty((0,1))
      return { 'updates': self.step, 'loss': float(np.mean(losses)),
        'acccuracy': float(th.mean(th.eq(logits <= 0.5, labels <= 0.5).float())),
        'entropy': float(th.mean(th.distributions.Bernoulli(logits).entropy())), 'logits': logits.detach(),
        'policy': float(th.sum(logits > 0.5)/len(labels)), "buffer": float(th.sum(logits <= 0.5)/len(labels)) 
      }

    def get_hparams(self, prefix=""):
        exclude = ['training', 'chi', 'device', 'parent', 'obs_dim', 'act_dim', 'rew_dim', 'optimizer', 'step']
        automatic = {f"{prefix}{k}" : v.__name__ if isinstance(v, type) else v for k, v in vars(self).items() if k not in exclude and not k.startswith('_')}
        return automatic