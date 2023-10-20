import numpy as np; import time; import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard.writer import SummaryWriter
from algorithm.logging import write_hyperparameters
import matplotlib.pyplot as plt; import pandas as pd; import scipy.stats as st

class EvaluationCallback(BaseCallback):
  """ Callback for evaluating an agent.
  :param model: The model to be evaluated^
  :param eval_envs: A dict containing environments for testing the current model.
  :param stop_on_reward: Whether to use early stopping. Defaults to True
  :param reward_threshold: The reward threshold to stop at."""
  def __init__(self, model: BaseAlgorithm, eval_envs: dict, stop_on_reward:float=None, record_video:bool=True, write_heatmaps:bool=True, run_test:bool=True):
    super(EvaluationCallback, self).__init__(); self.model = model; self.writer: SummaryWriter = self.model.writer
    self.eval_envs = eval_envs; self.record_video = record_video; self.write_heatmaps = write_heatmaps; self.run_test = run_test
    save_mean = lambda a: np.mean(a) if len(a) else np.nan; self.s = 0
    self.ep_mean = lambda key: save_mean([ep_info[key] for ep_info in self.model.ep_info_buffer]); self.stop_on_reward = stop_on_reward
    if stop_on_reward is not None: print(f"Stopping at {stop_on_reward}"); assert run_test, f"Can't stop on reward {stop_on_reward} without running test episodes"
    if record_video: assert run_test, f"Can't record video without running test episodes"

  # def init_callback(self, model: BaseAlgorithm) -> None:  print(f"Done Setup Learning in {(time.time_ns() - self.model.start_time)/1e+9}")
  # def _on_rollout_end(self) -> None: print(f"Collected rollout in {time.time()-self.start}") # add self.start = time.time() to _start

  def _on_rollout_start(self) -> None: # self.start = time.time()
    if self.writer == None: return 
    # Uncomment for early stopping based on 100-mean training return
    _sor = (self.ep_mean('reward_threshold') if self.stop_on_reward == 'VARY' else self.stop_on_reward)
    r = self.ep_mean('r'); self.model.progress_bar.postfix[0] = r; 
    self.model.progress_bar.update(self.model.num_timesteps-self.s); self.s = self.model.num_timesteps
    if _sor is not None and r >= _sor or not self.model.continue_training: self.model.continue_training = False
    if self.model.should_eval(): self.evaluate()


  def _on_step(self) -> bool: 
    """ Write timesteps to info & stop on reward threshold"""
    # print(f"Step took {time.time()-self.start}"); self.start = time.time()
    [info['episode'].update({'t': self.model.num_timesteps}) for info in self.locals['infos'] if info.get('episode')]
    return self.model.continue_training

  def _on_training_end(self) -> None: # No Early Stopping->Unkown, not reached (continue=True)->Failure, reached (stopped)->Success
    self.model.progress_bar.update(self.model.num_timesteps-self.s); self.s = self.model.num_timesteps
    if self.writer == None: return 
    status = 'STATUS_UNKNOWN' if not self.stop_on_reward else 'STATUS_FAILURE' if self.model.continue_training else 'STATUS_SUCCESS'
    metrics = self.evaluate(); write_hyperparameters(self.model, list(metrics.keys()), status)

  def prepare_ci(self, infos: dict, category=None, confidence:float=.95, write_raw=False) -> dict: 
    """ Computes the confidence interval := x(+/-)t*(s/√n) for a given survey of a data set. ref:
    https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/eval/meta_eval.py#L19
    :param infos: data dict in form {tag: {step: data, ...}}
    :param categroy: category string to write summary (also custom scalar headline), if None: deduced from info tag
    :param confidence: 
    :param wite_raw: bool flag wether to write single datapoints
    :return: Dict to be writen to tb in form: {tag: {step: value}} """
    summary, s = {}, self.model.num_timesteps; 
    factory = lambda c, t, m, ci: {f"{c}/{t}-mean":{s:m}, f"raw/{c}_{t}-lower":{s:m-ci}, f"raw/{c}_{t}-upper":{s:m+ci}}
    if write_raw: summary.update({f"raw/{category}_{tag}": info for tag, info in infos.items()})
    for tag, data in infos.items():
      d = np.array(list(data.values())).flatten(); mean = d.mean(); ci = st.t.ppf((1+confidence)/2, len(d)-1) * st.sem(d)
      if category == None: category,tag = tag.split('/')
      c = self.model._custom_scalars.get(category, {}); t = c.get(tag); update = False
      if not c: self.model._custom_scalars.update({category:{}}); c = self.model._custom_scalars.get(category); update = True
      if not t: c.update({tag: ['Margin', list(factory(category, tag, 0, 0).keys())]}); update = True
      if update: self.writer.add_custom_scalars(self.model._custom_scalars)
      summary.update(factory(category, tag, mean, ci))
    return summary


  def evaluate(self):
    """Run evaluation & write hyperparameters, results & video to tensorboard. Args:
        write_hp: Bool flag to use basic method for writing hyperparams for current evaluation, defaults to False
    Returns: metrics: A dict of evaluation metrics, can be used to write custom hparams """ 
    import time; start = time.time()
    step = self.model.num_timesteps
    if not self.writer: return []
    metrics = {k:v for label, env in self.eval_envs.items() for k, v in self.run_eval(env, label, step).items()}
    [self.writer.add_scalar(key, value, step) for key, value in metrics.items()]; self.writer.flush()
    
    summary = {}
    # Get infos from episodes & record rewards confidence intervals to summary 
    epdata = {name: {ep['t']: ep[key] for ep in self.model.ep_info_buffer} for key,name in self.model._naming.items()}
    summary.update(self.prepare_ci(epdata, category="rewards", write_raw=True))

    # Get Metrics from logger, record (float,int) as scalars, (ndarray) as confidence intervals
    logs = pd.json_normalize(self.model.logger.name_to_value, sep='/').to_dict(orient='records')
    if len(logs):
      summary.update({t: {step: v} for t,v in logs[0].items() if isinstance(v, (float,int))})
      summary.update(self.prepare_ci({t: {step: v} for t, v in logs[0].items() if isinstance(v, np.ndarray)}))
    
    #Write metrcis summary to tensorboard 
    [self.writer.add_scalar(tag, value, step) for tag,item in summary.items() for step,value in item.items()]
    self.model.eval()
    # print(f"Done Evaluating in {time.time()-start}") # Early stopping based on evaluation return
    # _sor = (self.ep_mean('reward_threshold') if self.stop_on_reward == 'VARY' else self.stop_on_reward)
    # r = metrics['validation_return']; self.model.progress_bar.postfix[0] = r; 
    # self.model.progress_bar.update(self.model.num_timesteps-self.s); self.s = self.model.num_timesteps
    # if _sor is not None and r >= _sor or not self.model.continue_training: self.model.continue_training = False
    self.writer.flush(); return metrics

  def run_eval(self, env, label: str, step: int):
    metrics = {}
    if self.run_test: 
      deterministic = False  # not env.get_attr('spec')[0].nondeterministic
      n_eval_episodes = 1    #if not deterministic else 100
      n_envs = env.num_envs; episode_rewards = []; episode_counts = np.zeros(n_envs, dtype="int")
      episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
      heatmap = np.zeros(env.envs[0].unwrapped.size)
      observations = env.reset(); states = None; episode_starts = np.ones((env.num_envs,), dtype=bool)
      while (episode_counts < episode_count_targets).any():
        heatmap[tuple(env.envs[0].unwrapped.getpos())] += 1
        actions, states = self.model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        new_observations, _, dones, infos = env.step(actions)
        for i in range(n_envs):
          if episode_counts[i] < episode_count_targets[i]:
            episode_starts[i] = dones[i]
            if dones[i] and "episode" in infos[i].keys():
              episode_rewards.append(infos[i]["episode"]["r"]); episode_counts[i] += 1
        observations = new_observations
      if self.write_heatmaps and env.envs[0].discrete: 
        fig = plt.figure(figsize=(10,10))
        plt.imshow(heatmap); plt.axis('off'); plt.colorbar(plt.pcolor(heatmap))
        self.writer.add_figure(f'exploration/{label}', fig, step)
        
        if hasattr(self.model, 'buffer') and len(self.model.buffer.observations):
          heatmap = np.zeros(env.envs[0].unwrapped.size)
          for sample in self.model.buffer.observations:
            heatmap[tuple(env.envs[0].unwrapped.getpos(board=sample.reshape(env.envs[0].unwrapped.size)))] += 1
          hmb = plt.figure(figsize=(10,10))
          plt.imshow(heatmap); plt.axis('off'); plt.colorbar(plt.pcolor(heatmap))
          self.writer.add_figure(f'buffer/{label}', hmb, step)

      metrics[f"rewards/{label}"] = np.mean(episode_rewards) #np.std(episode_rewards)
      if self.record_video: 
        frame_buffer =  env.envs[0].get_video()
        video = th.tensor(frame_buffer).unsqueeze(0).swapaxes(3,4).swapaxes(2,3)
        self.writer.add_video(label, video, global_step=step, fps=env.envs[0].metadata['render_fps'])

    # Create & write tringle heatmap plots
    self.writer.flush()
    return metrics
