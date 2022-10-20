""" Generic Algorithm Class extending BaseAlgorithm with features needed by the training pipeline """
import numpy as np; import pandas as pd
import torch as th; import scipy.stats as st
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy, obs_as_tensor as obs
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any, Dict, List, Optional, Type, Union
from tqdm import tqdm; import os
import platform; import stable_baselines3 as sb3; import gym
from .evaluation import EvaluationCallback

class TrainableAlgorithm(BaseAlgorithm):
  def __init__(self, envs:Optional[Dict[str,VecEnv]]=None, normalize:bool=False, policy:Union[str,Type[ActorCriticPolicy]]="MlpPolicy", path:Optional[str]=None, silent=False, **kwargs):
    """ :param env: The environment to learn from (if registered in Gym, can be str)
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...) defaults to MlpPolicy
    :param normalize: whether to use normalized observations, default: False
    :param path: (str) the log location for tensorboard (if None, no logging) """
    if envs: kwargs['env'] = envs['train']
    self.envs, self.normalize, self.path, self.silent = envs, normalize, path, silent
    self.eval_frequency, self.progress_bar = None, None
    super().__init__(policy=policy, verbose=0, **kwargs)
    
  def _setup_model(self) -> None:
    if self.normalize: self.env = VecNormalize(self.env)
    self._naming = {'l': 'length-100', 'r': 'return-100', 's': 'safety-100'}; self._custom_scalars = {}
    self.get_actions = lambda s: self.policy.get_distribution(s).distribution.probs.cpu().detach().numpy()
    self.heatmap_iterations = { 'policy': (lambda _, s, a, r: self.get_actions(obs(s, self.device))[0][a], (0,1)) }
    super(TrainableAlgorithm, self)._setup_model()
    self.writer, self._registered_ci = SummaryWriter(log_dir=self.path) if self.path else None, []
    if not self.silent: print("+-------------------------------------------------------+\n"\
      f"| System: {platform.platform()} |\n| Version: {platform.version()} |\n" \
      f"| GPU: {f'Enabled, version {th.version.cuda} on {th.cuda.get_device_name(0)}' if th.cuda.is_available() else'Disabled'} |\n"\
      f"| Python: {platform.python_version()} | PyTorch: {th.__version__} | Numpy: {np.__version__} |\n" \
      f"| Stable-Baselines3: {sb3.__version__} | Gym: {gym.__version__} | Seed: {self.seed:3d}    |\n"\
        "+-------------------------------------------------------+")

  #Helper functions for writing model or hyperparameters
  def _excluded_save_params(self) -> List[str]:
    """ Returns the names of the parameters that should be excluded from being saved by pickling. 
    E.g. replay buffers are skipped by default as they take up a lot of space.
    PyTorch variables should be excluded with this so they can be stored with ``th.save``.
    :return: List of parameters that should be excluded from being saved with pickle. """
    return super(TrainableAlgorithm, self)._excluded_save_params() + ['get_actions', 'heatmap_iterations', '_naming', '_custom_scalars', '_registered_ci', 'envs', 'writer', 'progress_bar', 'silent']

  def should_eval(self) -> bool: return self.eval_frequency is not None and self.num_timesteps % self.eval_frequency == 0  
  def should_log(self) -> bool: return self.log_frequency is not None and self.num_timesteps % self.log_frequency == 0 

  def learn(self, total_timesteps: int, stop_on_reward:float=None, eval_frequency=2048, log_frequency=256, **kwargs) -> "TrainableAlgorithm":
    """ Learn a policy
    :param total_timesteps: The total number of samples (env steps) to train on
    :param stop_on_reward: Threshold of the mean 100 episode return to terminate training.
    :param **kwargs: further aguments are passed to the parent classes 
    :return: the trained model """
    callback = EvaluationCallback(self, self.envs['test'], stop_on_reward=stop_on_reward); callback.evaluate() # Force Pre-training eval
    if 'callback' in kwargs: callback = CallbackList([kwargs.pop('callback'), callback])    
    alg = self.__class__.__name__; total = self.num_timesteps+total_timesteps; stepsize = self.n_steps * self.n_envs;
    if eval_frequency is not None: self.eval_frequency = eval_frequency * self.n_envs // stepsize * stepsize or eval_frequency * self.n_envs
    if log_frequency is not None: self.log_frequency = log_frequency * self.n_envs // stepsize * stepsize or log_frequency * self.n_envs
    hps = self.get_hparams(); hps.pop('seed'); hps.pop('num_timesteps');  # hp = f"(χ={self.chi}, κ={self.kappa}, ω={self.omega})" if alg=="DIRECT" else ""
    hyper = f"with: χ={hps.pop('chi')}, κ={hps.pop('kappa')}, ω={hps.pop('omega')}" if alg == "DIRECT" else ""
    disc = f"with {hps.pop('n_epochs')} / {hps.pop('discriminator_n_updates')} updates in {hps.pop('batch_size')} / {hps.pop('discriminator_batch_size')} batches [Policy/Discriminator] on {hps.pop('n_steps')} step rollouts" if alg == "DIRECT" else ""
    if not self.silent: print(f"Training {alg} {hyper} | in {hps.pop('env_name')} (x{hps.pop('n_envs')}) {disc}") 
    self.progress_bar = tqdm(total=total, unit="steps", postfix=[0,""], bar_format="{desc}[R: {postfix[0]:4.2f}][{bar}]({percentage:3.0f}%)[{n_fmt}/{total_fmt}@{rate_fmt}]") #desc=f"Training {alg}{hp}",
    self.progress_bar.update(self.num_timesteps);
    model = super(TrainableAlgorithm, self).learn(total_timesteps=total_timesteps, callback=callback, **kwargs)
    self.progress_bar.close()
    return model

  def train(self, **kwargs) -> None:
    # Update Progressbar 
    self.progress_bar.postfix[0] = np.mean([ep_info["r"] for ep_info in self.ep_info_buffer])
    if self.should_log(): self.progress_bar.update(self.log_frequency); #n_steps
    summary, step = {}, self.num_timesteps 

    super(TrainableAlgorithm, self).train(**kwargs) # Train PPO & Write Training Stats 
    if self.writer == None or not self.should_log(): return 

    # Get infos from episodes & record rewards confidence intervals to summary 
    epdata = {name: {ep['t']: ep[key] for ep in self.ep_info_buffer} for key,name in self._naming.items()}
    termination_reasons = self.env.get_attr('termination_reasons')
    [self.logger.record(str(r).replace('.','s/'), sum([env[r] for env in termination_reasons])) for r in termination_reasons[0]]
    summary.update(self.prepare_ci(epdata, category="rewards", write_raw=True))

    # Get Metrics from logger, record (float,int) as scalars, (ndarray) as confidence intervals
    metrics = pd.json_normalize(self.logger.name_to_value, sep='/').to_dict(orient='records')[0]
    summary.update({t: {step: v} for t,v in metrics.items() if isinstance(v, (float,int))})
    summary.update(self.prepare_ci({t: {step: v} for t, v in metrics.items() if isinstance(v, np.ndarray)}))
    
    #Write metrcis summary to tensorboard 
    [self.writer.add_scalar(tag, value, step) for tag,item in summary.items() for step,value in item.items()]
    # [self.writer.add_histogram(key, values, step) for key, values in metrics.items() if isinstance(values, th.Tensor)]
  
  def prepare_ci(self, infos: dict, category=None, confidence:float=.95, write_raw=False) -> Dict: 
    """ Computes the confidence interval := x(+/-)t*(s/√n) for a given survey of a data set. ref:
    https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/eval/meta_eval.py#L19
    :param infos: data dict in form {tag: {step: data, ...}}
    :param categroy: category string to write summary (also custom scalar headline), if None: deduced from info tag
    :param confidence: 
    :param wite_raw: bool flag wether to write single datapoints
    :return: Dict to be writen to tb in form: {tag: {step: value}} """
    summary, s = {}, self.num_timesteps; 
    factory = lambda c, t, m, ci: {f"{c}/{t}-mean":{s:m}, f"raw/{c}_{t}-lower":{s:m-ci}, f"raw/{c}_{t}-upper":{s:m+ci}}
    if write_raw: summary.update({f"raw/{category}_{tag}": info for tag, info in infos.items()})
    for tag, data in infos.items():
      d = np.array(list(data.values())).flatten(); mean = d.mean(); ci = st.t.ppf((1+confidence)/2, len(d)-1) * st.sem(d)
      if category == None: category,tag = tag.split('/')
      c = self._custom_scalars.get(category, {}); t = c.get(tag); update = False
      if not c: self._custom_scalars.update({category:{}}); c = self._custom_scalars.get(category); update = True
      if not t: c.update({tag: ['Margin', list(factory(category, tag, 0, 0).keys())]}); update = True
      if update: self.writer.add_custom_scalars(self._custom_scalars)
      summary.update(factory(category, tag, mean, ci))
    return summary

  def get_hparams(self):
    """ Fetches, filters & flattens own hyperparameters
    :return: Dict of Hyperparameters containing numbers and strings only """ 
    exclude = ['device','verbose','writer','tensorboard_log','start_time','rollout_buffer','eval_env']+\
      ['policy','policy_kwargs','policy_class','lr_schedule','sde_sample_freq','clip_range','clip_range_vf']+\
      ['env','observation_space','action_space','action_noise','ep_info_buffer','ep_success_buffer','target_kl']+\
      ['envs', 'path', 'progress_bar', 'disc_kwargs', 'buffer']
    hparams = pd.json_normalize(
      {k: v.__name__ if isinstance(v, type) else v.get_hparams() if hasattr(v, 'get_hparams') else 
          v for k,v in vars(self).items() if not(k in exclude or k.startswith('_'))
      }, sep="_").to_dict(orient='records')[0]
    hp_dis = {k:v for k,v in hparams.items() if isinstance(v, (int, bool))}
    hp_num = {k:v for k,v in hparams.items() if isinstance(v, (float))}
    hp_str = {k:str(v) for k,v in hparams.items() if isinstance(v, (str, list))}
    hp_mdd = {k:v for k,v in hparams.items() if isinstance(v, (dict, th.Tensor))} #list
    assert not len(hp_mdd), "Skipped writing hparams of multi-dimensional data"
    return {**hp_dis, **hp_num, **hp_str, **{'env_name': self.env.get_attr('env_name')[0]}}
  
  def save(self, name="/model/train", **kwargs) -> None: 
    kwargs['path'] = self.path + name; super(TrainableAlgorithm, self).save(**kwargs)

  @classmethod
  def load(cls, load, envs: Dict[str,VecEnv], path, **kwargs) -> "TrainableAlgorithm":
    kwargs['env'] = envs['train']; kwargs['envs'] = envs; load = f"{load}/model/train"
    assert os.path.exists(load+'.zip'), f"Attempting to load a model from {load} that does not exist"
    model = super(TrainableAlgorithm, cls).load(load, **kwargs)
    model.path = path; model.writer = SummaryWriter(log_dir=model.path)
    model.num_timesteps -= model.num_timesteps%(model.n_steps * model.n_envs)
    return model
