"""
Discriminator Network Architecture, Loss, Update, Batchsampling
Inspired by TODO: Link paper(s)
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

import numpy as np
import torch as th
from typing import Dict, Any, List

from gym.spaces.utils import flatdim as _dim
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.base_class import BaseAlgorithm

from . import DirectBuffer, DirectBufferSamples 

class Discriminator(th.nn.Module): #abc.ABC
    """The discriminator to use for DIRECT."""

    def __init__(
        self, batch_size: int, chi: float, hidden_size: List[int], 
        optimizer_class: th.optim.Optimizer = th.optim.Adam, optim_kwargs: Dict[str, Any] = {},
        activation:th.nn.Module=th.nn.ReLU, parent: BaseAlgorithm = None, #Sigmoid
        use_obs:bool=True, use_act:bool=True, use_ret:bool=True, 
    ):
        """Construct discriminator network.
        Args:
          env: the used environment (used to calculate the network dimensions
            & normalize batches of training data)
          batch_size: the batch size for training the discriminator.
          chi: mixture of real and discriminative rewards 
          hidden_size: Hidden size of the discriminator network
          optimizer_class: the optimizer class to use for training the disc
          optim_kwargs: additional parameters to be passed the optimizer

          ???? loadpath ??
          hidden_size: Hidden size of the MLP
          use_obs, use_act, use_ret: should the current observation/action/return 
            be included as an input to the MLP
          activation: activation to apply after hidden layers.

          Old:
           direct_net: a Torch module that that generates the reward
           TODO: Save/Load Params
        """
        super().__init__()
        self.activation = activation
        self.batch_size = batch_size
        self.chi = chi
        self.device = parent.device
        self.parent = parent

        self.obs_dim = _dim(self.parent.env.unwrapped.observation_space) * use_obs
        self.act_dim = _dim(self.parent.env.unwrapped.action_space) * use_act
        self.ret_dim = 1 * use_ret
        # self.parent.env.unwrapped.get_attr('reward_range') # TODO: not accessible, needed?
        self.hidden_size = hidden_size

        # TODO: check if Squeeze layer(missing) is needed -> reduced dim, not needed, dim(disc) == dim(ret)
        layer = lambda i,o: [th.nn.Linear(i,o), self.activation()]
        units =  [self.obs_dim + self.act_dim + self.ret_dim] + self.hidden_size + [1]
        layers = [l for i, o in zip(units, units[1:]) for l in layer(i,o)][:-1]
        self.net = th.nn.Sequential(*layers)

        self.optimizer_class = optimizer_class        
        # Use SGD for discriminator?? (https://github.com/soumith/ganhacks)
        # optim_kwargs.setdefault('lr', 3e-4) # Previous Default
        self.optimizer = self.optimizer_class(self.parameters(), **optim_kwargs)
        self.step = 0

    def forward(self, obs:th.Tensor, act:th.Tensor, ret:th.Tensor) -> th.Tensor:
        """Compute the discriminator's logits for a state-action-return sample.
        Args:
            obs: observation at time t.
            act: action taken at time t.
            ret: return received by the environmet for action at time t.
        Returns:
            preds: discriminative predictions for given activation (default: ReLU)
                    high (>0) -> current policy (generator)
                    low (<=0) -> imitation buffer (expert) """ 
        inputs = []
        if self.obs_dim: inputs.append(th.flatten(obs, 1))
        if self.act_dim: inputs.append(th.flatten(act, 1))
        if self.ret_dim: inputs.append(th.flatten(ret, 1))
        inputs_concat = th.cat(inputs, dim=1)
        return self.net(inputs_concat)

    def backward(self, obs:th.Tensor, act:th.Tensor, ret:th.Tensor, labels:th.Tensor) -> Dict[str, float]:
        """Perform a backward step, i.e. compute BCE-Loss, step optimizer, update parameters
        generates sigmoid-processed logits from batch: ]0;0.5]->pred_buffer ]0.5;1[->pred_policy
        Args: (parsed from the batch dict generated by `prepare_batch()`)
            obs: batch of observations 
            act: batch of actions 
            ret: batch of returns 
            labels: integer labels, with 0->buffer, 1->policy (c.f. prediction legend)
        Returns:
            stats: a dict containing training stats."""
        # Apply binary_cross_entropy_with_logits in two steps (1.sigmoid, 2.binary_cross_entropy) 
        logits = th.sigmoid(self.forward(obs, act, ret))
        # TODO: wasserstein loss ? https://developers.google.com/machine-learning/gan/loss
        loss = th.nn.functional.binary_cross_entropy(logits, labels.float())
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step(); self.step += 1

        return { # Return metrics
            "discriminator/step": self.step,
            "discriminator/loss": float(th.mean(loss)),
            "discriminator/acccuracy": float(th.mean(th.eq(logits <= 0.5, labels <= 0.5).float())),
            # entropy of the predicted label distribution, averaged equally across
            # both classes (if this drops then disc is very good or has given up)
            "discriminator/entropy": float(th.mean(th.distributions.Bernoulli(logits).entropy())),
            # policy / buffer: Portions of buffer and policy predictions 
            "discriminator/policy": float(th.sum(logits > 0.5)/len(labels)),
            "discriminator/buffer": float(th.sum(logits <= 0.5)/len(labels)),
            "discriminator/logits": logits.detach()
        }

    def reward(self, samples: DirectBufferSamples) -> np.ndarray:
        """Compute the discriminative rewards for a buffer of samples
        Normalization of DirectBufferSamples is handled in buffer.py
        Args: 
            samples: Buffer of experience
        Returns: 
            rewards: discriminative rewards, adapted for learning
                high  ->  imitation buffer
                low   ->  current policy 
        """
        obs = th.as_tensor(samples.observations, device=self.device)
        act = th.as_tensor(samples.actions, device=self.device)
        ret = th.as_tensor(samples.returns, device=self.device)

        reward = - th.nn.functional.logsigmoid(self.forward(obs=obs, act=act, ret=ret))
        assert reward.shape == ret.shape, "Reward should be of shape {}, but is {}!".format(ret.shape,reward.shape)
        return reward.cpu().numpy()

    def process(self, rollout: RolloutBuffer) -> DirectBuffer:
        """
        Takes a full rollout and calculates the discriminative rewards

        rollout: RolloutBuffer to extract information from
        returns: DirectBuffer: policy_buffer from current rollout
        """  
        with th.no_grad():
            policy_buffer = DirectBuffer.fromRollout(rollout, parent=self.parent)
            reward_shape = rollout.rewards.shape
            real_rewards = policy_buffer.swap_and_flatten(rollout.rewards.copy()) 
            samples = policy_buffer._get_samples(range(len(real_rewards)))
            disc_rewards = self.reward(samples) # TODO: check if to cpu missing
            # reshape = lambda t: t.cpu().numpy().reshape(real_rewards.shape)
            # disc_rewards = reshape(self.reward(samples))
        rewards = (self.chi * disc_rewards) + ((1-self.chi) * real_rewards)
        invert_shape = (reward_shape[1], reward_shape[0],  *reward_shape[2:])
        rollout.rewards = rewards.reshape(invert_shape).swapaxes(0, 1)
        assert policy_buffer.overwrites == 0, "Policy Buffer too small!"
        return policy_buffer
    
    def prepare_batch(self, policy_buffer: DirectBuffer, direct_buffer: DirectBuffer) -> Dict[str, th.Tensor]:
        """Build and return training batch for the next discriminator update.
        Args:
            direct_buffer: Buffer to sample expert.
            policy_buffer: Buffer to sample policy.
        Returns:
            batch: A batch of policy and buffer training samples.
        """
        buffer_samples: DirectBufferSamples = direct_buffer.sample(self.batch_size)
        policy_samples: DirectBufferSamples = policy_buffer.sample(self.batch_size)
        assert (len(buffer_samples.observations) == len(policy_samples.observations) == self.batch_size), "Missmatching batch sizes!"
        
        # Concatenate rollouts, and label each row as expert or generator (_gen_is_one)
        obs = th.cat([buffer_samples.observations, policy_samples.observations])
        act = th.cat([buffer_samples.actions, policy_samples.actions])
        ret = th.cat([buffer_samples.returns, policy_samples.returns])
        labels = th.cat([
            th.zeros_like(buffer_samples.returns, device=self.device, dtype=int), 
            th.ones_like(policy_samples.returns, device=self.device, dtype=int), 
        ])
        return { "obs": obs, "act": act, "ret": ret, "labels": labels}

    def train(self, policy_buffer: DirectBuffer, direct_buffer: DirectBuffer):
        """Perform a single discriminator update 
        (Generate Batch, perform a backward/training step & log results)
        Args: 
            direct_buffer: Buffer to sample expert.
            policy_buffer: Buffer to sample policy.
        Returns: 
            dict: Statistics for discriminator (e.g. loss, accuracy)."""
        batch = self.prepare_batch(policy_buffer=policy_buffer, direct_buffer=direct_buffer)
        stats = self.backward(**batch)
        return stats

    def get_hparams(self, prefix=""):
        exclude = ['training', 'chi', 'device', 'parent', 'obs_dim', 'act_dim', 'ret_dim', 'optimizer', 'step']
        automatic = {f"{prefix}{k}" : v.__name__ if isinstance(v, type) else v for k, v in vars(self).items() if k not in exclude and not k.startswith('_')}
        return automatic