from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from util.evaluation import evaluate
from stable_baselines3.common.base_class import BaseAlgorithm

from stable_baselines3.common.utils import safe_mean

from torch.utils.tensorboard.writer import SummaryWriter

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

from stable_baselines3.common.callbacks import BaseCallback

# NEW
"""
TODO:Add:

• Additional metrics from envs? calculate performance etc...
"""
from safety_env import factory
from stable_baselines3.common.env_util import make_vec_env

from .hparams import write_session_start_info, write_experiment, write_session_end_info
# END NEW

# TODO: rename?
class TestModel(BaseCallback):
  """
  Callback for evaluating an agent.

  :param env_name: Name of the safety environmnet to be evaluated ("test" stage) 
  :param env_spec: index of env config to be used (cf. safety_env/__init__.py)
  :param reward_threshold:  expected reward per episode to stop training. 
      No Early stopping for None. (in validation env from test stage)
  :param eval_freq: Evaluate the agent every ``eval_freq`` call (steps) of the callback.
      Optional, if none, set to n_steps (-> evaluate after every rollout)
  :param n_eval_episodes: The number of episodes to test the agent
  :param deterministic: Whether the evaluation should use a stochastic or deterministic actions.
  :param render: Whether to render or not the environment during evaluation
  :param write_ci: Wheather to additionally write the 95% confidence intervals of the metrics 
  :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
      wrapped with a Monitor wrapper)
  """

  def __init__(self, env_name: str, env_spec: int, reward_threshold: float = None, eval_freq: Optional[int] = None, n_eval_episodes: int = 1, 
    deterministic: bool = True, render: bool = False, write_video: bool = True, write_ci: bool = False,# -> render to tb?? / write_video
    warn: bool = True,
  ):
    super(TestModel, self).__init__()  # verbose=0 -> no logging, just write to tb
    self.envs = factory(env_name, make_vec_env, "test", env_spec)
    self.reward_threshold = reward_threshold
    self.eval_freq = eval_freq
    self.writer, self.hparam_infos = None,  None
    self.continue_training = True
    self._info_buffer = []

    # Passed to evaluation
    self.n_eval_episodes = n_eval_episodes
    self.deterministic = deterministic
    # self.render = render
    self.warn = warn

    self.write_ci = write_ci
    self.write_video = write_video
    self.metrics = ['reward', 'length']
    self.metric_suffix = ['_mean', '_lower', '_upper'] if write_ci else ['']
    self.metrics = [f'{metric}{suffix}'for metric in self.metrics for suffix in self.metric_suffix]
    self.best_mean_reward, self.last_mean_reward = -np.inf, -np.inf

    # TODO: was this behavior intended? evaluate validation env during training? 
    self.eval_env = self.envs['validation']
  
  def _init_callback(self) -> None:
    self.writer: SummaryWriter = self.model.writer
    hparams = self.model.get_hparams()
    hparam_infos = write_session_start_info(self.writer, hparams)
    metric_list = [f"metrics/{label}_{metric}" for metric in self.metrics for label in self.envs.keys()]
    write_experiment(self.writer, metric_list, hparam_infos)
    # if self.eval_freq is None: self.eval_freq = self.model.n_steps #+ 1 #max(self.model.n_steps // self.model.n_envs, 1) + 1
    # if self.eval_freq is None: self.eval_freq = max(self.model.n_steps // self.model.n_envs, 1)
    [self.writer.add_custom_scalars_marginchart(
        [f'{metric}{suffix}' for suffix in self.metric_suffix], category='metrics', title=metric
    ) for metric in self.metrics] if self.write_ci else []  
  
  def _info_on_done(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
    """
    Callback passed to the  ``evaluate_policy`` function
    in order to get the full 'info' content from the environemnt
    TODO: is this working? are all infos from env written to info buffer?
    """
    print("info on done")
    # print(self._info_buffer)
    if locals_["done"]: # Access specific infos by locals_["info"].get("is_success")
        self._info_buffer.append(locals_["info"])

  def _on_rollout_start(self) -> None:
    print("_on_rollout_start")

  def _on_rollout_end(self) -> None:
    # print(self._info_buffer)
    print("_on_rollout_ends")
    self.run_evaluation()

    # assert False

  def _on_step(self) -> bool:
    # return self.continue_training
    if self.eval_freq and self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      print(self.model.num_timesteps)
      print(self.eval_freq)
      # self.run_evaluation()
    return self.continue_training

  def _on_training_end(self) -> None: # No Early Stopping->Unkown, not reached (continue=True)->Failure, reached (stopped)->Success
    status = 'STATUS_UNKNOWN' if self.reward_threshold is None else 'STATUS_FAILURE' if self.continue_training else 'STATUS_SUCCESS'
    write_session_end_info(self.writer, status)

  def run_evaluation(self) -> None:
    # Sync training and eval env if there is VecNormalize
    # TODO AssertionError: Training and eval env are not of the same type <stable_baselines3.common.vec_env.vec_normalize.VecNormalize object at 0x7f9944d9b6d0> != <stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv object at 0x7f98314748b0>
    # assert isinstance(self.training_env, type(self.envs['validation'])), f"Training and eval env are not of the same type {self.training_env} != {self.envs['validation']}"
    if self.model.get_vec_normalize_env() is not None:
      try: sync_envs_normalization(self.training_env, self.eval_env)
      except AttributeError: raise AssertionError("Training and eval env are not wrapped the same way, see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback and warning above.")

      # Reset info buffer
      self._info_buffer = []

      # NEW
      # return_episode_rewards=True    
      # #TODO: if self.write_ci   
      # 
      # if self.write_video:
      #     print("TODO")


      # TODO: pass to run_eval args
      # step = self.model.num_timesteps
      # metrics = {k:v for label, env in self.envs.items() for k, v in run_eval(self.model, env, self.model.writer, label, step).items()}
      # # Write hp metrics manual (adapted from torch.utils.tensorboard.writer.add_hparams)    
      # # [model.writer.file_writer.add_summary(file) for file in to_hp_log(hparams, metrics)] #, discretes
      # # [model.writer.add_scalar(k, v, step) for k, v in metrics.items()]
      # model.writer.flush()
      # print("Done evaluating")

      #         .. warning::

      #   When using multiple environments, each call to  ``env.step()``
      #   will effectively correspond to ``n_envs`` steps.
      #   To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

        # END NEW
      # TODO: pass ep_info_on_done?
      episode_rewards, episode_lengths = evaluate_policy(
        self.model,
        self.eval_env,
        n_eval_episodes=self.n_eval_episodes,
        render=self.render,
        deterministic=self.deterministic,
        return_episode_rewards=True,
        warn=self.warn,
        callback=self._log_success_callback,
      )


      mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
      mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
      self.last_mean_reward = mean_reward

      # Add to current Logger
      # TODO Record video to tb 
      # TODO: use eval tags for hparams
      # TODO: write to tb at timestep {self.num_timesteps}}, vars: mean_reward, std_reward, mean_ep_length, std_ep_length
      self.logger.record("eval/mean_reward", float(mean_reward))
      self.logger.record("eval/mean_ep_length", mean_ep_length)

      if len(self._info_buffer) > 0:
          print(self._info_buffer)

      # Dump log so the evaluation results are printed with the correct timestep
      self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
      self.logger.dump(self.num_timesteps)

      if mean_reward > self.best_mean_reward:
          self.best_mean_reward = mean_reward
          # Trigger callback if needed
          if self.reward_threshold is not None:
              self.continue_training = bool(self.best_mean_reward < self.reward_threshold)
                