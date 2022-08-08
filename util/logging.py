import numpy as np; import pandas as pd; import torch as th
import scipy.stats as st
from typing import List, Dict

from torch.utils.tensorboard.writer import SummaryWriter

from torch.utils.tensorboard.summary import hparams as to_hp_log, custom_scalars

def prepare_ci(naming: dict, writer: SummaryWriter): [writer.add_custom_scalars_marginchart(list(_get_ci([0,0], k).keys())) for k in naming.values()]

def _get_ci(category:str, tag:str, data:np.ndarray, confidence: float = 0.95) -> Dict:
  """ Computes the confidence interval := x(+/-)t*(s/√n) for a given survey of a data set. ref:
  (x: sample mean, t: t-value that corresponds to the confidence level, s: sample standard deviation, n: sample size)
  - https://stackoverflow.com/a/15034143/1601580, https://www.geeksforgeeks.org/how-to-calculate-confidence-intervals-in-python/
  - https://github.com/WangYueFt/rfs/blob/f8c837ba93c62dd0ac68a2f4019c619aa86b8421/eval/meta_eval.py#L19 """
  x=data.mean(); n=len(data)-1; t=st.t.ppf((1+confidence)/2,n); sn=st.sem(data)
  return {f"{tag}-mean": x, f"{tag}-lower": x-t*sn, f"{tag}-upper": x+t*sn}

def write_episode_infos(info_buffer: List, naming: Dict, writer: SummaryWriter, step:int):
  # data = {f"rewards/100-mean-{name}": [episode[key] for episode in info_buffer] for key,name in naming.items()}
  ci_data = {name: {ep['t']: ep[key] for ep in info_buffer} for key,name in naming.items()}
  # [writer.add_scalar(
  prepare_ci(writer, ci_data)
  [writer.add_scalar(f"rewards_raw/{name}", episode[key], episode['t']) for episode in info_buffer for key, name in naming.items()]  # Write plain Episode Infos 
  # [writer.add_scalar(tag,data,step) for key,name in naming.items() for tag,data in _get_ci([episode[key] for episode in info_buffer], f"rewards/100-mean-{name}").items()]
  # [writer.add_scalar(tag,data,step) for key,name in naming.items() for tag,data in _get_ci([episode[key] for episode in info_buffer], f"rewards/100-mean-{name}").items()]

def write_train_metrics(metrics: Dict, writer: SummaryWriter, step: int):
  metrics = pd.json_normalize(metrics, sep='/').to_dict(orient='records')[0]
  [writer.add_scalar(key, value, step) for key, value in metrics.items() if isinstance(value, (float,int))]
  # [_write_ci(key, values, step, writer) for key, values in metrics.items() if isinstance(values, np.ndarray)]
  # TODO: register ci for array shaped train metrics

  [writer.add_histogram(key, values, step) for key, values in metrics.items() if isinstance(values, th.Tensor)]
  writer.flush()
  
  print("np")
  [print(key) for  key, values in metrics.items() if isinstance(values, np.ndarray)]
  # print(metrics['rewards/environment'])
  # print(len(metrics['rewards/environment']))

  # print([ep_info["r"] for ep_info in self.ep_info_buffer])
  # print(metrics["rewards/environment"])
  # print(metrics["rewards/discriminator"])
  # TODO: alternative to write hists: write ci's, todo: make sure, mulitline plots are registered    
  #TODO: cumulate disc & env rewards in similar manner to additional safety metrics
  # eval.write_ci(real_rewards, self.writer, "rewards", "environment", self.num_timesteps, self.n_steps, 0.2)
  # eval.write_ci(disc_rewards, self.writer, "rewards", "discriminator", self.num_timesteps, self.n_steps, 0.2)
  # Alt handling:       
  # self.logger.record("rewards/enironment", float(real_rewards.mean()))
  # self.logger.record("rewards/discriminator", float(disc_rewards.mean()))