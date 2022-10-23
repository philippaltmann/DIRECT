import os; from os import path; import itertools; from parse import parse; from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
import pandas as pd; import numpy as np; import scipy.stats as st
from safety_env import factory, make, env_name, env_spec; from algorithm import *
from run import env_conf

def extract_model(exp, run):
  algorithm, seed = eval(exp['algorithm']), int(run.name)
  envs = factory(*env_conf(exp['env']), n_train=1, n_test=1)[0]
  envs['train'].seed(seed); [env.seed(seed) for _,env in envs['test'].items()]
  model = algorithm.load(load=run.path, seed=seed, envs=envs, path=None, silent=True, device='cpu')
  return model

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

  print(f"Scanning for {alg if alg else 'algorithms'} in {base}")  # First layer: Algorithms
  if alg: experiments = [{'algorithm': alg, 'path':f'{base}/{alg}'}]
  else: experiments = [{'algorithm': a.name, 'path': a.path} for a in subdirs(base) if a.name in ALGS]

  print(f"Scanning for {env if env else 'environments'} in {base}")  # Second layer: Environments
  if env: experiments = [{**exp, 'env': env, 'path': f'{exp["path"]}/{env}'} for exp in experiments]
  else: experiments = [{**exp, 'env': e.name, 'path': e} for exp in tqdm(experiments) for e in subdirs(exp['path'])]

  print(f"Scanning for hyperparameters in {base}")  # Third layer: Hyperparameters & number of runs
  hp = lambda name: dict(zip(['chi','omega','kappa'], parse('{:.1f}_{:.1f}_{:d}', name) or []))
  experiments = [{ **exp, 'path': e.path, **hp(e.name), 'runs': len(subdirs(e)) }
    for exp in tqdm(experiments) if os.path.isdir(exp['path']) for e in subdirs(exp['path'])]
  if alg=='DIRECT': experiments = sorted(experiments, key=lambda exp: (exp['kappa'],exp['omega']))

  print(f"Loading experiment logfiles from {metrics} [{sum([exp['runs'] for exp in experiments])* len(metrics)}]")  # Fourth layer: tb files   
  progressbar = tqdm(total=sum([exp['runs'] for exp in experiments])* len(metrics))
  data_buffer = {}

  def fetch_data(exp, run_path, name, key):
    # Load data from csv if possible
    if path.isfile(f'{run_path}/{name}.csv'): return pd.read_csv(f'{run_path}/{name}.csv').set_index('Step')

    # Use buffered Event Accumulator if already open
    if log := data_buffer.get(run_path):
      extract_args = {'columns': ['Time', 'Step', 'Data'], 'index': 'Step', 'exclude': ['Time']}
      data = pd.DataFrame.from_records(log.Scalars(key), **extract_args)
      data = data.loc[~data.index.duplicated(keep='first')] # Remove duplicate indexes
      if dump_csv: data.to_csv(f'{run_path}/{name}.csv')
      return data    

    data_buffer.update({run_path: EA(run_path).Reload()})
    return fetch_data(exp, run_path, name, key)
  
  def extract_data(exp, run, name, key):
    progressbar.update()
    if name == 'Model': return key
    data = fetch_data(exp, run.path, name, key)
    if baseline is None: return data
    path = run.path.replace(base, baseline).replace(exp['env'], 'TrainingDense' if 'Dense' in exp['env'] else 'TrainingSparse')
    prev = fetch_data(exp, path, name, key)
    data.index = data.index - prev.index[-1] # Norm by last baseline index
    return data
  
  # Process given experiments
  process_data = lambda exp, name, key: [ extract_data(exp, run, name, key) for run in subdirs(exp['path']) ] 
  fetch_models = lambda exp: [ extract_model(exp, run) for run in subdirs(exp['path']) ] 
  experiments = [{**exp, 'data': { name: process_data(exp, *scalar) for name, scalar in metrics }, 'models': fetch_models(exp)} for exp in experiments]
  progressbar.close()

  return experiments


def group_experiments(experiments, groupby=['algorithm', 'env'], label_excl=[], mergeon=None): #merge=None
  # Graphical helpers for titles, labels
  forms = {'algorithm':'{}', 'env':'{}', 'chi': 'Χ: {:.1f}', 'omega':  'ω: {:.2f}', 'kappa': 'κ: {:d}'}
  label = lambda exp, excl=label_excl: ' '.join([f.format(exp[key]) for key, f in forms.items() if key in exp and key not in groupby + excl])
  title = lambda exp: ' '.join([f.format(exp[key]) for key, f in forms.items() if key in exp and key in groupby])
  def hue(index): index[0] += 1; return 360 / index[1] * index[0] - 180/(index[1] * index[0]); #32; 180

  # Create product of all occuances of specified groups, zip with group titles & add size and a counter of visited group items
  def merge(experiments, key):
    values = {exp[key]:'' for exp in experiments}.keys(); get = lambda d,k,*r: get(d[k],*r) if len(r) else d[k]
    merge = lambda val, *k: [item for exp in experiments for item in get(exp,*k) if exp[key]==val] 
    extract = lambda val, k, *r, p=[]: {k: extract(val, *r, p=[*p,k])} if len(r) else {k: merge(val, *p, k) } 
    data = lambda val: {k:v for key in experiments[0]['data'] for k,v in extract(val, 'data', key)['data'].items()}
    return[{key:val,  **extract(val, 'models'), 'data': data(val)} for val in values]
  if mergeon is not None: experiments = merge(experiments, mergeon)
  options = list(itertools.product(*[ list(dict.fromkeys([exp[group] for exp in experiments])) for group in groupby ]))
  ingroup = lambda experiment, group: all([experiment[k] == v for k,v in zip(groupby, group)])
  options = list(zip(options, [[ title(exp) for exp in experiments if ingroup(exp,group)] for group in options ]))
  options = [(group, [0, len(titles)], titles[0]) for group, titles in options]
  getgraph = lambda exp, index: { 'label': label(exp), 'data': exp['data'], 'models': exp['models'], 'hue': hue(index)} 
  return [{'title': title, 'graphs': [ getgraph(exp, index) for exp in experiments if ingroup(exp, group) ] } for group, index, title in options ]


def calculate_metrics(plots, metrics):
  """Given a set of experiments and metrics, applies metric calulation
  :param plots: list of dicts containing plot information and graphs with raw data
  :param metrics: Dicts of metric names and calculation processes 
  """
  def process(metric, proc, plot):
    graphs = [ { **graph, 'data': proc(graph['data'][metric], graph['models']) } for graph in plot['graphs']]
    if metric == 'Heatmap':
      return [ { 'title': f"{plot['title']} | {graph['label']} | {key} ", 'data': data, 'metric': metric} 
        for graph in graphs for key, data in graph['data'].items() ]
    return [{ **plot, 'graphs': graphs, 'metric': metric}]
  return [ result for metric in metrics for plot in plots for result in process(*metric, plot)]


def process_ci(data, models):
  # Helper to claculate confidence interval
  ci = lambda d, confidence=0.95: st.t.ppf((1+confidence)/2, len(d)-1) * st.sem(d)
  stop = models[0].envs['train'].get_attr('reward_threshold')[0]
  upper = models[0].envs['train'].get_attr('env')[0].spec.reward_threshold

  # Prepare Data (fill until highest index)
  steps = [d.index[-1] for d in data]; maxsteps = np.max(steps)
  for d in data: d.at[maxsteps] = float(d.tail(1)['Data'])
  data = pd.concat(data, axis=1, ignore_index=False, sort=True).bfill()
  
  # Mean 1..n | CI 1..n..1
  mean, h = data.mean(axis=1), data.apply(ci, axis=1)
  ci = pd.concat([mean+h, (mean-h).iloc[::-1]]).clip(upper=upper)
  return (mean, ci, stop)


def process_steps(data, models): return ([d.index[-1] for d in data], 10e5)


iterate = lambda model, envs, func: [ func(env, k,i) for env in envs for k,i in model.heatmap_iterations.items() ]
heatmap = lambda model, envs: iterate(model, envs, lambda env, k,i: env.envs[0].iterate(i[0]))
metadata = lambda model, envs: iterate(model, envs, lambda env, k,i: (f'{k.capitalize()} Env-{env_spec(env).id[-1]}', i[1]))
make_env = lambda model, spec: make(env_name(model.envs['train']), spec, seed=model.seed)

def process_heatmap(specs, models):
  setting = list(zip(models, [[make_env(model, s) for s in spec] for model,spec in zip(models,specs)]))
  return { k:(d,a) for (k,a),d in zip(metadata(*setting[0]), np.moveaxis(np.array([heatmap(*s) for s in setting]), 0, -1))} 


def process_eval(specs, models, deterministic=True):
  from stable_baselines3.common.evaluation import evaluate_policy
  data = [(model, make_env(model, spec)) for model,spec in zip(models,specs)]
  termination_reasons = data[0][1].get_attr('termination_reasons')[0]
  def callback(g,l): 
    if 'episode' in g['info']: termination_reasons[g['info']['extra_observations']['termination_reason']] += 1
  if deterministic: eval = [evaluate_policy(*args, n_eval_episodes=1, callback=callback)[0] for args in data ]
  else: eval = [r for args in data for r in evaluate_policy(*args, n_eval_episodes=100, deterministic=False, return_episode_rewards=True, callback=callback)[0]]

  return (eval, data[0][1].get_attr('reward_threshold')[0], termination_reasons)
  
