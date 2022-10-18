import os; from os import path; import itertools; from parse import parse; from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
import pandas as pd; import numpy as np; import scipy.stats as st
from safety_env import factory, make; from algorithm import *

def extract_model(exp, run, env_spec=0):
  algorithm, seed= eval(exp['algorithm']), int(run.name)
  envs = factory(seed, exp['env'], spec=env_spec, n_train=1, n_test=1)
  model = algorithm.load(load=run.path, seed=seed, envs=envs, path=None, silent=True)
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

  def extract_data(exp, run, name, key):
    progressbar.update()
    if name == 'Model': return key

    # Load data from csv if possible
    if path.isfile(f'{run.path}/{name}.csv'): return pd.read_csv(f'{run.path}/{name}.csv').set_index('Step')

    # Helper to filter double indexes & arguments for extraction
    rm_dpl_idx = lambda data: data.loc[~data.index.duplicated(keep='first')]
    extract_args = {'columns': ['Time', 'Step', 'Data'], 'index': 'Step', 'exclude': ['Time']}

    log = data_buffer.get(run.path)
    if log: return rm_dpl_idx(pd.DataFrame.from_records(log.Scalars(key), **extract_args))
    else: data_buffer.update({run.path: EA(run.path).Reload()})
    return extract_data(exp, run, name, key)
  
  # Process given experiments
  process_data = lambda exp, name, key: [ extract_data(exp, run, name, key) for run in subdirs(exp['path']) ] 
  fetch_models = lambda exp, spec = 0: [ extract_model(exp, run, spec) for run in subdirs(exp['path']) ] 
  experiments = [{**exp, 'data': { name: process_data(exp, *scalar) for name, scalar in metrics }, 'models': fetch_models(exp)} for exp in experiments]
  progressbar.close()

  names = list(zip(*list(zip(*metrics))[1]))[0] # Extract scalar names from metrics list 
  dump_experiment = lambda data, runs: [ df.to_csv(f'{r.path}/{m}.csv') for m, d in data for r, df in zip(runs, d) if m != 'Model' ]
  if dump_csv: [dump_experiment(zip(names,exp['data'].values()), subdirs(exp['path'])) for exp in experiments]

  return experiments


def group_experiments(experiments, groupby=['algorithm', 'env']):
  # Graphical helpers for titles, labels
  forms = {'algorithm':'{}', 'env':'{}', 'chi': 'Χ: {:.1f}', 'omega':  'ω: {:.2f}', 'kappa': 'κ: {:d}'}
  label = lambda exp, excl=[]: ' '.join([f.format(exp[key]) for key, f in forms.items() if key in exp and key not in groupby + excl])
  title = lambda exp: ' '.join([f.format(exp[key]) for key, f in forms.items() if key in exp and key in groupby])
  def hue(index): index[0] += 1; return 360 / index[1] * index[0] - 180/index[1]; #32; 

  # Create product of all occuances of specified groups, zip with group titles & add size and a counter of visited group items
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

  # Prepare Data (fill until highest index)
  steps = [d.index[-1] for d in data]; maxsteps = np.max(steps)
  for d in data: d.at[maxsteps] = float(d.tail(1)['Data'])
  data = pd.concat(data, axis=1, ignore_index=False, sort=True).bfill()
  
  # Mean 1..n | CI 1..n..1
  mean, h = data.mean(axis=1), data.apply(ci, axis=1)
  ci = pd.concat([mean+h, (mean-h).iloc[::-1]])
  return (mean, ci)


def process_steps(data, models): return ([model.num_timesteps for model in models])


def process_heatmap(models):
  envs = lambda model: model.envs['test'].items()
  iters = lambda model: model.heatmap_iterations.items()
  iterate = lambda n, env, k, iter: env.envs[0].iterate(iter[0]) #[1,2,3]#('data', iter[1])#
  heatmap = lambda model: np.array([ iterate(*e, *i) for e in envs(model) for i in iters(model) ])
  metadata = lambda n, env, k, iter: (f'{k.capitalize()} {n.capitalize()}', iter[1])#todo: append later - {graph["label"]} 
  return { key: (data, args) for (key, args), data in zip( 
    [ metadata(*e, *i) for e in envs(models[0]) for i in iters(models[0]) ], # Titles & Args [no. heatmaps]
    np.moveaxis(np.array([heatmap(model) for model in models]), 0, -1) # Heatmap Data [models,heatmaps,...]->[heatmaps,...,models] 
  )} 
