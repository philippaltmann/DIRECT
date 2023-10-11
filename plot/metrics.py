import os; from os import path; import itertools; from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
import pandas as pd; import numpy as np; import scipy.stats as st; import re
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym; from hyphi_gym import named, Monitor
from stable_baselines3.common.monitor import Monitor as SBM
from benchmark import *

def extract_model(exp, run):
  if exp['explorer'] not in ['Random', 'LOAD']: 
    algorithm, seed = eval(exp['explorer']), int(run.name)
  else: algorithm, seed = eval(exp['algorithm']), int(run.name)
  pre = algorithm.load(load=run.path, phase='pre', seed=seed, envs=[exp['env']], path=None, device='cpu', silent=True)
  train = algorithm.load(load=run.path, seed=seed, envs=[exp['env']], path=None, device='cpu', silent=True)
  return {'pre': pre, 'train': train}

def fetch_experiments(base='./results', alg=None, env=None, metrics=[], dump_csv=False, baseline=None):
  """Loads and structures all tb log files Given:
  :param base_path (str):
  :param env (optional): the environment to load 
  :param alg (optional): the algorithm to load 
  :param metrics: list of (Name, Tag) tuples of metrics to load
  :param save_csv: save loaded experiments to csv
  Returns: list with dicts of experiments 
  """
  # Helper to fetch all relevant folders 
  subdirs = lambda dir: [d for d in os.scandir(dir) if d.is_dir()]

  print(f"Scanning for {env if env else 'environments'} in {base}")  # First layer: Environments
  if env: experiments = [{'env': e.name, 'path': e.path} for e in subdirs(base) if env == e.name] # in
  else: experiments = [{'env': e.name, 'path': e.path} for e in subdirs(base) if e.name != 'plots']

  print(f"Scanning for {alg if alg else 'algorithms'} in {base}")  # Second layer: Algorithms
  if alg: experiments = [{**exp, 'algorithm': alg, 'path': a} for exp in tqdm(experiments) for a in subdirs(exp['path']) if alg == a.name]
  else: experiments = [{**exp, 'algorithm': a.name, 'path': a} for exp in tqdm(experiments) for a in subdirs(exp['path']) if any([n in ALGS for n in a.name.split('-')])]
  experiments = [{**e, 'algorithm': e['algorithm'].split('-')[-1], 'explorer': e['algorithm'].split('-')[0] if len(e['algorithm'].split('-'))>1 else 'Random'} for e in tqdm(experiments)]


  # Third Layer: Count Runs / fetch tb files
  print(f"Scanning for hyperparameters in {base}")  # Third layer: Hyperparameters & number of runs
  experiments = [{ **e, 'runs': len(subdirs(e['path'])) } for e in tqdm(experiments) if os.path.isdir(e['path'])]
  # experiments = [{ **exp, 'path': e.path,  'method': e.name, 'runs': len(subdirs(e)) } for exp in tqdm(experiments) if os.path.isdir(exp['path']) for e in subdirs(exp['path'])] # With hp

  progressbar = tqdm(total=sum([exp['runs'] for exp in experiments])* len(metrics))
  data_buffer = {}

  def fetch_data(exp, run_path, name, key):
    # Load data from csv if possible
    if path.isfile(f'{run_path}/{name}.csv'): return pd.read_csv(f'{run_path}/{name}.csv').set_index('Step')
    if 'Model' in key: return f'{run_path}/{name}.csv' if dump_csv else None
    if 'Buffer' in name and 'GRASP' not in run_path: return None

    # Use buffered Event Accumulator if already open
    if log := data_buffer.get(run_path):
      extract_args = {'columns': ['Time', 'Step', 'Data'], 'index': 'Step', 'exclude': ['Time']}
      data = pd.DataFrame.from_records([(s.wall_time, s.step, s.value) for s in log.Scalars(key)], **extract_args)
      data = data.loc[~data.index.duplicated(keep='first')] # Remove duplicate indexes
      # if exp['explorer'] in ['GAIN', 'GRASP']: data.iloc[0] = -100
      if name == 'Training': data.iloc[0] = np.NaN  # Remove initial zero
      if 'Buffer' in name: data = pd.concat([pd.DataFrame([0], columns=['Data']), data]) / 2048; data.index.name = 'Step'
      if dump_csv: data.to_csv(f'{run_path}/{name}.csv')
      return data
    tb_path = run_path+("/explore/" if '-pre' in name else "/train/")
    data_buffer.update({run_path: EA(tb_path).Reload()})
    return fetch_data(exp, run_path, name, key)
  
  def extract_data(exp, run, name, key):#42*2048*4  #48*2048*4 #52 458752
    progressbar.update()
    data = fetch_data(exp, run.path, name, key)
    
    if name == 'Training': 
      if data.index[0] > 0: data.index = data.index - data.index[0]
      if exp['explorer'] in ['GAIN', 'GRASP']: data.index = data.index + (42*2048*4)/8

    # Load Buffer
    if 'Buffer' in name and exp['explorer'] in ['GRASP']: 
      data = (data, np.load(f'{run.path}/buffer.npy'))

    return data
  
  # Process given experiments
  finished = lambda dir: [d for d in subdirs(dir) if len(subdirs(d))] # subdirs(d)[0].name == 'model'
  process_data = lambda exp, name, key: [ extract_data(exp, run, name, key) for run in finished(exp['path']) ] 
  fetch_models = lambda exp: [ extract_model(exp, run) for run in finished(exp['path'])] 
  experiments = [{
      **exp, 'data': { name: process_data(exp, name, scalar) for name, scalar in metrics }, 'models': fetch_models(exp)
    } for exp in experiments]
  progressbar.close()
  return experiments


def group_experiments(experiments, groupby=['env']): #merge=None
  # Graphical helpers for titles, labels
  title = lambda exp: ' '.join([exp[key] for key in ['algorithm', 'explorer', 'env'] if key in exp and key in groupby])
  options = list(itertools.product(*[ list(dict.fromkeys([exp[group] for exp in experiments])) for group in groupby ]))
  ingroup = lambda experiment, group: all([experiment[k] == v for k,v in zip(groupby, group)])
  options = list(zip(options, [[ title(exp) for exp in experiments if ingroup(exp,group)] for group in options ]))
  options = [(group, [0, len(titles)], titles[0]) for group, titles in options]
  getgraph = lambda exp, index: { 'label': f"{exp['algorithm']}-{exp['explorer']}", **exp}  #'models': exp['models'], 'env': exp['env']
  return [{'title': title, **dict(zip(groupby,group)), 'graphs': [ getgraph(exp, index) for exp in experiments if ingroup(exp, group) ] } for group, index, title in options ]


def calculate_metrics(plots, metrics):
  """Given a set of experiments and metrics, applies metric calulation
  :param plots: list of dicts containing plot information and graphs with raw data
  :param metrics: Dicts of metric names and calculation processes 
  """
  for plot in plots: 
    env = SBM(Monitor(gym.make(**named(plot['env']))))
    seeds = list(set([model['train'].seed for graph in plot['graphs'] for model in graph['models']]))
    reward_threshold, reward_range = env.unwrapped.reward_threshold, env.unwrapped.reward_range
    # if reward_threshold is not None and 'VARY' in reward_threshold: 
    if reward_threshold == 'VARY': 
      def iterget(key, seed): env.reset(seed=seed); return getattr(env,key)
      reward_threshold = np.array([iterget('reward_threshold', s) for _ in range(int(100/len(seeds))) for s in seeds]).mean()
      reward_range = tuple(np.array([iterget('reward_range', s) for _ in range(int(100/len(seeds))) for s in seeds]).mean(0))
    env.info = {'seeds': seeds, 'reward_range':reward_range, 'reward_threshold': reward_threshold}
    plot['env'] = env
  return [ result for metric, process in metrics for plot in plots for result in process(metric, plot)]


def prepare_ci(data, bounds):
  # Helper to claculate confidence interval
  ci = lambda d, confidence=0.95: st.t.ppf((1+confidence)/2, len(d)-1) * st.sem(d)
  
  # Prepare Data (fill until highest index)
  steps = [d.index[-1] for d in data]; maxsteps = np.max(steps)
  for d in data: d.at[maxsteps, 'Data'] = float(d.tail(1)['Data'].iloc[0])
  data = pd.concat(data, axis=1, ignore_index=False, sort=True).bfill()
  
  # Mean 1..n | CI 1..n..1
  mean, h = data.mean(axis=1), data.apply(ci, axis=1)
  ci = pd.concat([mean-h, (mean+h).iloc[::-1]]).clip(*bounds[::-1])
  return (mean, ci)


def prepare_eval(env, model, callback=lambda g,l: None, **kwargs):
  kwargs = {'n_eval_episodes':100, 'deterministic':False, 'return_episode_rewards':True, **kwargs} 
  env.reset(seed=model.seed); # print("Running evaluation")
  results = evaluate_policy(model=model, env=env, callback=callback, **kwargs)
  return results


def prepare_heatmap(env, model, data=None, buffer=None, progress=None):
  if isinstance(data, pd.DataFrame): return data
  else: path = data
  contpos = lambda obs: tuple(int(o) for o in (obs[2:4] + np.array(env.unwrapped.size) / 2 - 0.5).round())

  heatmap = np.zeros(env.unwrapped.size)
  def update_heatmap(pos): heatmap[pos] += 1 
  def callback(l,_): 
    if env.unwrapped.step_scale > 1: pos = contpos(l['observations'][0])  # Continuous
    else: pos = tuple(l['env'].envs[0].unwrapped.getpos()) # DISCRETE
    update_heatmap(pos)
  if buffer is not None: 
    if env.unwrapped.step_scale > 1: [update_heatmap(contpos(obs)) for obs in buffer]  # Continuous
    else: [update_heatmap(tuple(env.unwrapped.getpos(board=obs.reshape(env.unwrapped.size)))) for obs in buffer]
      
  else: prepare_eval(env, model, callback=callback)
  heatmap = pd.DataFrame(heatmap.ravel(), columns=['Obs']); heatmap.index.name = 'Step'
  if path is not None and buffer is None: heatmap.to_csv(path)
  if progress is not None: progress.update(1)
  return heatmap


def process_return(metric, plot):
  return [{ **plot, 'graphs':  [{ 
    **graph, 'data': prepare_ci(graph['data'][metric], plot['env'].info['reward_range']) 
  } for graph in plot['graphs']], 'metric': metric}]


def process_buffer(metric, plot): 
  env = plot['env']; env.unwrapped.explore = '-pre' in metric; phase = 'pre' if env.unwrapped.explore else 'train'; 
  graph = [graph for graph in plot['graphs'] if graph["explorer"] == "GRASP"][0]
  momentum, buffers = [list(d) for d in zip(*graph['data'][metric])]
  return [{
    'momentum':{ **plot, 'graphs':  [{ **graph, 'data': prepare_ci(momentum, plot['env'].info['reward_range'])}], 'metric': metric},
    'heatmap': { **plot, 'graphs':  [
        { **graph, 'data': pd.concat([ prepare_heatmap(env, model[phase], buffer=buffer) for buffer, model in zip(buffers, graph['models'])], axis=1)['Obs'].sum(axis=1)}
      ], 'metric': metric}, 'metric': metric,
  }]


def process_eval(metric, plot):
  env = plot['env']; env.unwrapped.explore = False; phase = 'pre' if '-pre' in metric else 'train'; 
  def _eval(data, model, progress):
    if isinstance(data, pd.DataFrame): return data
    R = pd.DataFrame(prepare_eval(env, model)[0], columns=['Return']); R.index.name = 'Step'
    if data is not None: R.to_csv(data)
    progress.update(1)
    return R
  print("Running evaluations")
  with tqdm(total=len(plot['graphs'])*len(plot['graphs'][0]['models'])) as progress:
    return [{ **plot, 'graphs':  [{ 
      **graph, 'data': pd.concat([_eval(data, model[phase], progress) for data, model in zip(graph['data'][metric], graph['models'])])['Return']
    } for graph in plot['graphs']], 'metric': metric}]


def process_heatmap(metric, plot):
  env = plot['env']; env.unwrapped.explore = '-pre' in metric; phase = 'pre' if env.unwrapped.explore else 'train'; 
  print("Processing Heatmaps")
  with tqdm(total=len(plot['graphs'])*len(plot['graphs'][0]['models'])) as progress:
    return [{ **plot, 'graphs':  [{ 
      **graph, 'data': pd.concat([ prepare_heatmap(
        env, model[phase], data=data, progress=progress
      ) for data, model in zip(graph['data'][metric], graph['models'])], axis=1)['Obs'].sum(axis=1)
    } for graph in plot['graphs']], 'metric': metric}]

