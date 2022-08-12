import numpy as np; import torch as th; from typing import Any, Dict
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard.writer import SummaryWriter
from util.logging import write_hyperparameters

class EvaluationCallback(BaseCallback):
  """ Callback for evaluating an agent.
  :param model: The model to be evaluated
  :param eval_envs: A dict containing environments for testing the current model.
  :param stop_on_reward: Whether to use early stopping. Defaults to True
  :param reward_threshold: The reward threshold to stop at."""
  def __init__(self, model: BaseAlgorithm, eval_envs: dict, stop_on_reward:float=None):
    super(EvaluationCallback, self).__init__() 
    self.model = model; self.eval_envs = eval_envs; self.writer: SummaryWriter = self.model.writer
    self.stop_on_reward, self.continue_training = stop_on_reward, True

  def _on_rollout_end(self) -> None:
    mean_return = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
    if (self.stop_on_reward and mean_return >= self.stop_on_reward) or not self.continue_training: self.continue_training = False
    self.evaluate()

  def _on_step(self) -> bool: 
    """ Write timesteps to info & stop on reward threshold"""
    [info['episode'].update({'t': self.model.num_timesteps}) for info in self.locals['infos'] if info.get('episode')]
    return self.continue_training if self.stop_on_reward else True

  def _on_training_end(self) -> None: # No Early Stopping->Unkown, not reached (continue=True)->Failure, reached (stopped)->Success
    status = 'STATUS_UNKNOWN' if not self.stop_on_reward else 'STATUS_FAILURE' if self.continue_training else 'STATUS_SUCCESS'
    metrics = self.evaluate(); write_hyperparameters(self.model, list(metrics.keys()), status)

  def evaluate(self):
    """Run evaluation & write hyperparameters, results & video to tensorboard. Args:
        write_hp: Bool flag to use basic method for writing hyperparams for current evaluation, defaults to False
    Returns: metrics: A dict of evaluation metrics, can be used to write custom hparams """ 
    step = self.model.num_timesteps
    metrics = {k:v for label, env in self.eval_envs.items() for k, v in self.run_eval(env, label, step).items()}
    [self.writer.add_scalar(key, value, step) for key, value in metrics.items()]; self.writer.flush()    
    return metrics

  def run_eval(self, env, label: str, step: int, eval_kwargs: Dict[str, Any]={}, write_video:bool=True):
    video_buffer, FPS, metrics = [], 10, {} # Move video frames from buffer to tensor, unsqueeze & clear buffer
    def retreive(buffer): entries = buffer.copy(); buffer.clear(); return th.tensor(np.array(entries)).unsqueeze(0)
    record_video = lambda locals, _: video_buffer.append(locals['env'].render(mode='rgb_array'))
    eval_kwargs.setdefault('n_eval_episodes', 1); eval_kwargs.setdefault('deterministic', True)  # Deterministic policy + env -> det behavior?
    metrics[f"metrics/{label}_reward"] = evaluate_policy(self.model, env, callback=record_video, **eval_kwargs)[0]
    metrics[f'metrics/{label}_performance'] = env.env_method('get_performance')[0]
    if write_video: self.writer.add_video(label, retreive(video_buffer), step, FPS) 
    return metrics
