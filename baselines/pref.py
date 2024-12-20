import numpy as np
from baselines import PPO 
from .reward_model import RewardModel
from .buffers import EntReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


class PrefPPO(PPO):
  """A preference-based extension to PPO according to: 
  https://arxiv.org/pdf/1706.03741 / https://arxiv.org/pdf/2111.03026"""

  def __init__( self, envs, **kwargs): # Shift
    super().__init__(envs, **kwargs); base_pref = {'max_feed': self.n_steps, 'feed_type': 0, 'num_interaction': self.n_steps*self.n_envs}
    if 'Grid' in envs[0]: self.pref_kwargs = { **base_pref, 'model_kwargs': {'size_segment': 32, 'mb_size':32}}
    elif 'Maze' in envs[0]: self.pref_kwargs = { **base_pref, 'model_kwargs': {'size_segment': 32, 'mb_size':32}}
    elif 'Fetch' in envs[0]: self.pref_kwargs = {**base_pref, 'model_kwargs': {'size_segment': 64, 'mb_size': 32}}
    else: assert False, f'Pref-Parameters for {envs[0]} not set'    

  def learn(self, load_path=None, total_timesteps:int=2e6, callback=[], reset_num_timesteps:bool=True, **kwargs) -> "PrefPPO":    
    self.pref_kwargs['unsuper_step'] = total_timesteps // 4  # 1/4 Unsupervised pre-training
    callback = CallbackList([PrefCallback(load_path=load_path, **self.pref_kwargs), *callback])
    return super(PrefPPO, self).learn(total_timesteps, reset_num_timesteps=reset_num_timesteps, callback=callback, **kwargs)


class PrefCallback(BaseCallback):
  """A Callback for Preference-Based Algorithms
  Adapted from https://github.com/rll-research/BPref"""

  def __init__( self, load_path = None, model_kwargs={},
    num_interaction: int = 16000, feed_type: int = 0, max_feed: int = 1400, # Reward Learning 
    unsuper_step: int = 32000, unsuper_n_epochs: int = 50,                  # Unsupervised Pre-training
  ):
    super().__init__(verbose=0)
    self.thres_interaction = num_interaction; self.num_interactions = 0 
    self.feed_type = feed_type; self.max_feed = max_feed; self.total_feed = 0; self.labeled_feedback = 0
    self.unsuper_step = unsuper_step; self.unsuper_n_epochs = unsuper_n_epochs
    save_mean = lambda a: np.mean(a) if len(a) else np.nan
    self.ep_mean = lambda key: save_mean([ep_info[key] for ep_info in self.model.ep_info_buffer])
    self.load_path = load_path; self.model_kwargs = model_kwargs

  def _init_callback(self) -> None:
    self.model.rollout_buffer.real_rewards = np.zeros_like(self.model.rollout_buffer.rewards)
    # instantiating the reward model TODO: handle case without reward model
    self.reward_model = RewardModel(self.training_env, device=self.model.device, **self.model_kwargs)
    if self.load_path is not None: self.reward_model.load(self.load_path)

    if self.unsuper_step > 0: # Init Buffer for unsupervised pre-training
      self.unsuper_buffer = EntReplayBuffer(self.unsuper_step+100, self.training_env, self.model.device)


  def step_unsuper(self, obs, act, infos) -> np.ndarray:
    self.unsuper_buffer.add_obs(obs)
    state_entropy = self.unsuper_buffer.compute_state_entropy(obs, norm=True)
    pred_reward = state_entropy.reshape(-1).data.cpu().numpy()
    return pred_reward


  def step_reward(self, obs, act, infos) -> np.ndarray:
    # StartPref num_env x (obs+act) -> num_env x 1 x (obs+act) -> r_hat
    act = np.reshape(act, (act.shape[0], act.shape[-1]))
    obsact = np.concatenate((obs, act), axis=-1)
    pred_reward = self.reward_model.r_hat_batch(obsact).reshape(-1)
    self.num_interactions += self.training_env.num_envs
    return pred_reward


  def learn_reward(self) -> None:
    # Update teacher parameters and generate queries (0-uniform|1-disagreement)
    self.reward_model.update_margin(self.ep_mean('r'))
    labeled_queries = self.reward_model.sample(disagreement=self.feed_type==1) 
    self.total_feed += self.reward_model.mb_size; self.labeled_feedback += labeled_queries
    
    # Train Reward Model, reset interaction & log results 
    rew_acc = self.reward_model.train(); self.num_interactions = 0
    mean = np.mean([r[0] for r in self.reward_model.targets])
    print(f"{self.num_timesteps}: Acc: {rew_acc}, {labeled_queries} new queries, {self.total_feed} total feedback [{self.reward_model.buffer_index}] {len(self.reward_model.inputs)}. R: {mean}")
    self.model.logger.record("rewards/model_accuracy", rew_acc)
    self.model.logger.record("rewards/total_feed", self.total_feed)
    self.model.logger.record("rewards/labeled_feedback", self.labeled_feedback)


  def _on_step(self) -> bool: 
    l = lambda name: self.locals.get(name)

    # Get Observations actions and environment rewards, save real_reward for training discriminator
    obs, act, real_rewards, infos = self.model._last_obs, l('actions')[:, np.newaxis], l('rewards'), l('infos')
    self.model.rollout_buffer.real_rewards[self.model.rollout_buffer.pos] = np.array(real_rewards).copy()

    if self.num_timesteps < self.unsuper_step: pred_reward = self.step_unsuper(obs, act, infos)
    elif self.reward_model is not None: pred_reward = self.step_reward(obs, act, infos)
    
    for i, (info, r_hat) in enumerate(zip(infos, pred_reward)):
      self.locals['rewards'][i] = r_hat # Overwrite rewards & add samples to buffer
      if ep := info.get('episode'): self.reward_model.add_data(ep) 

    if (self.num_interactions >= self.thres_interaction and self.total_feed < self.max_feed) or (self.num_timesteps == self.unsuper_step):
      self.learn_reward()
    return True

  def _on_training_end(self) -> None: self.reward_model.save(self.model.path)
