
from typing import Any, Callable, Dict, List, Optional, Type, Union
import gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


"""
Adaptation of FlattenExtractor (stable_baselines3/common/torch_layers.py)
Fixing MutliDiscrete Environments 
"""
class SafetyFeatureExtractor(BaseFeaturesExtractor):
    """ Extracting Flat, 1-Hot-encoded features from n-dimensional DiscreteSpaces """
    def __init__(self, observation_space: gym.spaces.MultiDiscrete):
        self.num_classes = observation_space.nvec.flat[0]
        num_flatobs = gym.spaces.utils.flatdim(observation_space)
        super(SafetyFeatureExtractor, self).__init__(observation_space, num_flatobs * self.num_classes)
        self.flatten = th.nn.Flatten()
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(th.nn.functional.one_hot(observations, num_classes=self.num_classes).float())


"""Usage: Pass class to policy parameter"""
class SafteyActorCriticPolicy(ActorCriticPolicy):
  """
  Extension to stable baselines ActorCriticPolicy class (stable_baselines3/common/policies.py)
  Fixing handling of MultiDiscrete environments like this SafetyEnv 
  """
  def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, lr_schedule: Callable[[float], float], 
    features_extractor_class: Type[BaseFeaturesExtractor] = SafetyFeatureExtractor, **policy_kwargs):
    assert isinstance(observation_space, gym.spaces.MultiDiscrete), \
      "This Policy is designed for SafetyEnvs that provide MultiDiscrete obeservations"
    super(SafteyActorCriticPolicy, self).__init__(observation_space, action_space, 
      lr_schedule, features_extractor_class=features_extractor_class, **policy_kwargs)

  """Preprocess the observation using SafetyEnvExtractor above."""
  def extract_features(self, obs: th.Tensor) -> th.Tensor: return self.features_extractor(obs)
