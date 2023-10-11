""" Generic Algorithm Class extending BaseAlgorithm with features needed by the training pipeline """
import numpy as np; import pandas as pd; import random; import torch as th; 
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy, obs_as_tensor as obs
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from torch.utils.tensorboard.writer import SummaryWriter; import stable_baselines3 as sb3; 
from typing import Optional, Type, Union; from tqdm import tqdm; 
import os; import psutil; import platform
import gymnasium as gym; from environment import factory
from .evaluation import EvaluationCallback


class TrainableAlgorithm(BaseAlgorithm):
  def __init__(self, envs:list[str]=None, normalize:bool=False, policy:Union[str,Type[ActorCriticPolicy]]="MlpPolicy", path:Optional[str]=None, seed=None, silent=False, stop_on_reward=False, explore=False, log_name=None, **kwargs):
    """ :param env: The environment to learn from (if registered in Gym, can be str)
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...) defaults to MlpPolicy
    :param normalize: whether to use normalized observations, default: False
    :param stop_on_reward: bool for ealry stopping, defaults to False. 
    :param explore: sets enviornment to explore mode, default False
    :param log_name: optional custom folder name for logging
    :param path: (str) the log location for tensorboard (if None, no logging) """
    _path = lambda seed: f"{path}/{envs[0]}/{log_name or str(self.__class__.__name__)}/{seed}"
    gen_seed = lambda s=random.randint(0, 999): s if not os.path.isdir(_path(s)) else gen_seed()
    if seed is None: seed = gen_seed()
    self.path = _path(seed) if path is not None else None; self.eval_frequency, self.progress_bar = None, None
    if envs is not None: self.envs = factory(envs, seed=seed, explore=explore); 
    self.explore = explore; self.stop_on_reward = stop_on_reward and not explore
    self.normalize, self.silent, self.continue_training = normalize, silent, True; 
    super().__init__(policy=policy, seed=seed, verbose=0, env=self.envs['train'], **kwargs)
    
  def _setup_model(self) -> None:
    if self.normalize: self.env = VecNormalize(self.env)
    self._naming = {'l': 'length-100', 'r': 'return-100'}; self._custom_scalars = {} #, 's': 'safety-100'
    self.get_actions = lambda s: self.policy.get_distribution(obs(np.expand_dims(s, axis=0), self.device)).distribution.probs   
    self.heatmap_iterations = { # Deterministic policy heatmaps
      'action': (lambda _,s,a,r: self.policy.predict(s.flat, deterministic=True)[0] == a, (0,1)),
      # Prob distributions (coelation of porb index and action number might be misalligned)
      'policy': (lambda _, s, a, r: self.get_actions(s).cpu().detach().numpy()[0][a], (0,1))}
    super(TrainableAlgorithm, self)._setup_model(); stage = '/explore' if self.explore else '/train'
    self.writer, self._registered_ci = SummaryWriter(self.path + stage) if self.path and not self.silent else None, [] 
    dev = ["CPU", *(['MPS'] if th.backends.mps.is_available() else []), *(['CUDA'] if th.cuda.is_available() else [])]
    dev[dev.index(str(self.device).upper())] = '*' + dev[dev.index(str(self.device).upper())]
    mem = round(psutil.virtual_memory().total / (1024.0 **3))
    if not self.silent and not self.explore: print("+----------------------------------------------------+\n"\
      f"| System: {platform.platform()}                 |\n" \
      f"| Devices: [ {' | '.join(dev)} ] | {os.cpu_count()} Cores | {mem} GB        |\n"\
      f"| Python: {platform.python_version()} | PyTorch: {th.__version__} | Numpy: {np.__version__}    |\n" \
      f"| Stable-Baselines3: {sb3.__version__} | Gym: {gym.__version__} | Seed: {self.seed:3d} |\n"\
        "+----------------------------------------------------+")

  #Helper functions for writing model or hyperparameters
  def _excluded_save_params(self) -> list[str]:
    """ Returns the names of the parameters that should be excluded from being saved by pickling. 
    E.g. replay buffers are skipped by default as they take up a lot of space.
    PyTorch variables should be excluded with this so they can be stored with ``th.save``.
    :return: List of parameters that should be excluded from being saved with pickle. """
    return super(TrainableAlgorithm, self)._excluded_save_params() + ['get_actions', 'heatmap_iterations', '_naming', '_custom_scalars', '_registered_ci', 'envs', 'writer', 'progress_bar', 'silent']

  def should_eval(self) -> bool: return self.eval_frequency is not None and self.num_timesteps % self.eval_frequency == 0  

  def learn(self, total_timesteps: int, eval_frequency=8192, eval_kwargs={}, **kwargs) -> "TrainableAlgorithm":
    """ Learn a policy
    :param total_timesteps: The total number of samples (env steps) to train on
    :param eval_kwargs: stop_on_reward: Threshold of the mean 100 episode return to terminate training., record_video:bool=True, write_heatmaps:bool=True, run_test:bool=True
    :param **kwargs: further aguments are passed to the parent classes 
    :return: the trained model """
    _rt = [e.unwrapped.reward_threshold for e in self.env.envs]
    stop_on_reward = ('VARY' if _rt[0] == 'VARY' else sum(_rt)/self.n_envs) if self.stop_on_reward else None
    callback = EvaluationCallback(self, self.envs['test'], stop_on_reward=stop_on_reward, **eval_kwargs); 
    if 'callback' in kwargs: callback = CallbackList([kwargs.pop('callback'), callback])    
    total = self.num_timesteps+total_timesteps
    if eval_frequency is not None: self.eval_frequency = eval_frequency * self.n_envs
    hps = self.get_hparams(); hps.pop('seed'); hps.pop('num_timesteps');  
    self.progress_bar = tqdm(total=total, unit="steps", postfix=[0,""], bar_format="{desc}[R: {postfix[0]:4.2f}][{bar}]({percentage:3.0f}%)[{n_fmt}/{total_fmt}@{rate_fmt}]") 
    self.progress_bar.update(self.num_timesteps); 
    model = super(TrainableAlgorithm, self).learn(total_timesteps=total_timesteps, callback=callback, **kwargs)
    self.progress_bar.close()
    return model

  def train(self, **kwargs) -> None:
    if not self.continue_training: return
    self.progress_bar.postfix[0] = np.mean([ep_info["r"] for ep_info in self.ep_info_buffer])
    if self.should_eval(): self.progress_bar.update(self.eval_frequency); #n_steps
    super(TrainableAlgorithm, self).train(**kwargs) # Train PPO & Write Training Stats 
  
  def get_hparams(self):
    """ Fetches, filters & flattens own hyperparameters
    :return: Dict of Hyperparameters containing numbers and strings only """ 
    exclude = ['device','verbose','writer','tensorboard_log','start_time','rollout_buffer','eval_env']+\
      ['policy','policy_kwargs','policy_class','lr_schedule','sde_sample_freq','clip_range','clip_range_vf']+\
      ['env','observation_space','action_space','action_noise','ep_info_buffer','ep_success_buffer','target_kl']+\
      ['envs', 'path', 'progress_bar', 'disc_kwargs', 'buffer','log_ent_coef']
    hparams = pd.json_normalize(
      {k: v.__name__ if isinstance(v, type) else v.get_hparams() if hasattr(v, 'get_hparams') else 
          v for k,v in vars(self).items() if not(k in exclude or k.startswith('_'))
      }, sep="_").to_dict(orient='records')[0]
    hp_dis = {k:v for k,v in hparams.items() if isinstance(v, (int, bool))}
    hp_num = {k:v for k,v in hparams.items() if isinstance(v, (float))}
    hp_str = {k:str(v) for k,v in hparams.items() if isinstance(v, (str, list))}
    hp_mdd = {k:v for k,v in hparams.items() if isinstance(v, (dict, th.Tensor))} #list
    assert not len(hp_mdd), "Skipped writing hparams of multi-dimensional data"
    return {**hp_dis, **hp_num, **hp_str, **{'env_name': self.env.unwrapped.envs[0].unwrapped.name}}
  
  def save(self, name="/model/train", **kwargs) -> None: 
    kwargs['path'] = self.path + name; super(TrainableAlgorithm, self).save(**kwargs)

  @classmethod
  def load(cls, load, phase='train', device: Union[th.device, str] = "auto", **kwargs) -> "TrainableAlgorithm":
    assert os.path.exists(f"{load}/model/{phase}.zip"), f"Attempting to load a model from {load} that does not exist"
    data, params, pytorch_variables = load_from_zip_file(f"{load}/model/{phase}", device=device)
    assert data is not None and params is not None, "No data or params found in the saved file"
    model = cls(device=device, _init_setup_model=False, **kwargs)
    model.__dict__.update(data); model.__dict__.update(kwargs); model._setup_model()
    model.set_parameters(params, exact_match=True, device=device)
    if pytorch_variables is not None: [recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)
        for name in pytorch_variables if pytorch_variables[name] is not None]
    if model.use_sde: model.policy.reset_noise() 
    model.num_timesteps -= model.num_timesteps%(model.n_steps * model.n_envs)
    model.envs = factory(model.envs, seed=model.seed, explore=model.explore); 
    model.env = model.envs['train']
    if model.normalize: model.env = VecNormalize(model.env)
    return model
