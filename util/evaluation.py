"""
Adapted from stable_baselines3.common.evaluation
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch as th
from typing import Any, List, Dict
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.tensorboard.summary import hparams as to_hp_log

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped



###############
### Eval v2 ###
###############
##############
# EVALUATION #
##############

import numpy as np; import torch as th
from typing import Any, List, Dict
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.tensorboard.summary import hparams as to_hp_log

# from ai_safety_gridworlds.environments.shared.safety_game import Actions
# safety_game.py

from util.hparams import *


"""
TODO: validation (testing) env not deterministic (random level 1|2)
TODO: log additional safety env metrics
-> ai_safety_gridworlds/environments/distributional_shift.py 
-> choose level 1 static on env creation?? 
# TODO environment heatmap reward viz

    # TODO chech those helpers ? 
    writer.add_custom_scalars_multilinechart(["list", "of", "keys"]); writer.flush()
    writer.add_custom_scalars_marginchart(["list", "of", "keys"]); writer.flush()

TODO? Non-Deterministic mode \w ci & num_iterations?
    
"""

# def evaluate(algorithm: BaseAlgorithm, env: str, spec: int, path: str = None, model: BaseAlgorithm = None):
# TODO: copy to callback?
def evaluate(envs: dict, model: BaseAlgorithm = None, path: str = None, algorithm: BaseAlgorithm = None, write_hp: bool = False):
  """Run evaluation & write hyperparameters, results & video to tensorboard. Args:
      algorithm: Class of the algorithm to be evaluated 
      env: Name of the safety environmnet to be evaluated
      spec: index of env config to be used (cf. safety_env/__init__.py)
      path: path to a saved model to be loaded 
      model: model to be evaluated
      write_hp: Bool flag to use basic method for writing hyperparams for current evaluation, defaults to False
  Returns:  metrics: A dict of evaluation metrics, can be used to write custom hparams """ 
  if model is None:
    assert path is not None, "Either the model or the path to a saved model needs to be specified!"
    model = algorithm.load(path, env=envs["validation"]) #, print_system_info=True #TODO: aktualisieren mit neuer load funktion
    assert isinstance(model, algorithm), "The Model needs to be of the given class"
  step = model.num_timesteps
  metrics = {k:v for label, env in envs.items() for k, v in run_eval(model, env, model.writer, label, step).items()}
  if write_hp: 
    assert False, "Not Implemented"
    # [model.writer.file_writer.add_summary(file) for file in to_hp_log(get_hparams(model), metrics)] 
  else: [model.writer.add_scalar(key, value, step) for key, value in metrics.items()]
  model.writer.flush()    
  return metrics

def run_eval(model: BaseAlgorithm, env, writer: SummaryWriter, label: str, step: int, eval_kwargs: Dict[str, Any]={}, write_video:bool=True):
  # Helpers for recording video
  video_buffer, FPS, metrics = [], 10, {} # 25
  # Move video frames from buffer to tensor, unsqueeze & clear buffer
  def retreive(buffer): entries = buffer.copy(); buffer.clear(); return th.tensor(np.array(entries)).unsqueeze(0)
  record_video = lambda locals, _: video_buffer.append(locals['env'].render(mode='rgb_array'))
  
  # Deterministic policy + env -> det behavior?
  eval_kwargs.setdefault('n_eval_episodes', 1)
  eval_kwargs.setdefault('deterministic', True)
  metrics[f"metrics/{label}_reward"] = evaluate_policy(model, env, callback=record_video, **eval_kwargs)[0]
  metrics[f'metrics/{label}_performance'] = env.env_method('get_performance')[0]
  if write_video: writer.add_video(label, retreive(video_buffer), step, FPS) 
  # writer.add_scalar(key, metrics, step); writer.flush() 

  # f = lambda state: model.policy.get_distribution(th.tensor(state).to(model.device))
  # f = lambda state: model.policy.forward(th.tensor(state).to(model.device))
  # def eval_policy(obs):
  #   # action, state= model.predict(obs, state=None, episode_start=None, deterministic=True)
  #   model.policy.set_training_mode(False)
  #   observation, vectorized_env = model.policy.obs_to_tensor(obs)
  #   actions = th.tensor(np.arange(model.action_space.n)).to(model.device)
  #   with th.no_grad():
  #     # actions = model.policy._predict(observation)
      
  #     values, log_prob, entropy = model.policy.evaluate_actions(observation,actions)
  #   print(actions)
  #   print(values)
  #   print(log_prob)
  #   print(entropy)
  #   print(Actions.DOWN)
  #   print(log_prob[Actions.DOWN])

  #   v = model.policy.predict_values(observation)
  #   print(v)
  #   # actions, values, log_probs = model.policy.forward(obs_tensor)
  #   # print(actions)
  #   # print(values)
  #   # print(log_probs)
  #   assert False
  # # f = lambda state: model.predict(state)
  # # print(env.reset())
  # test_result = env.env_method('iterate', eval_policy)
  # # print(test_result)
  # assert False, "TODO: Plot env iteration"


  return metrics #{key: metrics}


###############
### Eval v1 ###
###############
def evaluate(algorithm: BaseAlgorithm, env: str, spec: int, path: str = None, model: BaseAlgorithm = None):
    """Run evaluation & write hyperparameters, results & video to tensorboard 
    Args:
        algorithm: Class of the algorithm to be evaluated 
        env: Name of the safety environmnet to be evaluated
        spec: index of env config to be used (cf. safety_env/__init__.py)
        path: path to a saved model to be loaded 
        model: model to be evaluated
    Returns:
        ?batch: A batch of policy and buffer training samples.
    """ 
    envs = factory(env, make_vec_env, "test", spec) #TODO: move env generation out? (-> test.py & callback.py)
    if model is None:
        assert path is not None, "Either the model or the path to a saved model needs to be specified!"
        model = algorithm.load(path, env=envs["validation"]) #, print_system_info=True
    assert isinstance(model, algorithm), "The Model needs to be of the given class"
    step = model.num_timesteps

    #TODO create speperate writer?
    metrics = {k:v for label, env in envs.items() for k, v in run_eval(model, env, model.writer, label, step).items()}
    # Write hp metrics manual (adapted from torch.utils.tensorboard.writer.add_hparams)    
    # TODO: need to integrate \w custom hp writer? (this creates full hp set of files)
    [model.writer.file_writer.add_summary(file) for file in to_hp_log(model.get_hparams(), metrics)] #, discretes
    # [model.writer.add_scalar(k, v, step) for k, v in metrics.items()]
    model.writer.flush()
    # print("Done evaluating")

def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
