import os; from os import path; import itertools; from parse import parse; from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
import pandas as pd; import numpy as np; import scipy.stats as st
from baselines import *
from safety_env import factory;

# TODO: pass / load env spec
def extract_model(exp, run, env_spec=0):
  algorithm, seed= eval(exp['algorithm']), int(run.name)
  envs = factory(seed, exp['env'], spec=env_spec, n_train=1, n_test=1)
  model = algorithm.load(load=run.path, seed=seed, envs=envs, path=None, silent=True)
  return model

def fetch_experiments(base='./results', alg=None, env=None, metrics=[], dump_csv=False):
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
  else: experiments = [{'algorithm': a.name, 'path': a} for a in subdirs(base) if a.name in ALGS]

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
    # Load model if needed
    if name == 'Model': return extract_model(exp, run, key)

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
  experiments = [{**exp, 'data': { name: process_data(exp, *scalar) for name, scalar in metrics } } for exp in experiments]
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
  def hue(index): index[0] += 1; return 360 / index[1] * index[0] - 32; 
  # [print(title(exp) + " | " + label(exp)) for exp in experiments]

  # Create product of all occuances of specified groups, zip with group titles & add size and a counter of visited group items
  options = list(itertools.product(*[ list(dict.fromkeys([exp[group] for exp in experiments])) for group in groupby ]))
  ingroup = lambda experiment, group: all([experiment[k] == v for k,v in zip(groupby, group)])
  options = list(zip(options, [[ title(exp) for exp in experiments if ingroup(exp,group)] for group in options ]))
  options = [(group, [0, len(titles)], titles[0]) for group, titles in options]
  # options = list(zip(options, [ (0, len([ 1 for exp in experiments if ingroup(exp,group)])) for group in options ]))
  # print(f"{args.groupby} ∈ {options}"); [print(group) for group in options]

  # getdata = lambda exp, index: { 'label': label(exp), 'exp': exp, 'hue': hue(index) }
  getgraph = lambda exp, index: { 'label': label(exp), 'data': exp['data'], 'hue': hue(index) }
  return [{'title': title, 'graphs': 
      [ getgraph(exp, index) for exp in experiments if ingroup(exp, group) ]
    } for group, index, title in options ]


def calculate_metrics(plots, metrics):
  """Given a set of experiments and metrics, applies metric calulation
  :param plots: list of dicts containing plot information and graphs with raw data
  :param metrics: Dicts of metric names and calculation processes 
  """
  # process = lambda graph: { **graph, 'data': {name: metrics[name](data) for name, data in graph['data'].items()}}
  # return [{ **plot, 'graphs': [process(graph) for graph in plot['graphs']] } for plot in plots]

  process = lambda name, proc, graph: { **graph, 'data': proc(graph['data'][name]) }
  # process = lambda name, proc, graph: { 'data': proc(graph['data'][name]) }
  # process = lambda name, proc, graph: print(f"{name}: {proc}")#{ **graph, 'data': proc(graph['data'][name]) }
  iterate = lambda metric, graphs: [ process(*metric, graph) for graph in graphs]
  return [ { **plot, 'metric': metric[0], 'graphs': iterate(metric, plot['graphs']) }
    for metric in metrics for plot in plots]


def process_ci(data):
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


def process_steps(data): return [d.index[-1] for d in data]

def process_heatmap(models):
  heatmap_data = lambda model: { f'{key.capitalize()} {label.capitalize()}':(env.envs[0].iterate(f), args) 
    for label, env in model.envs['test'].items()  for key, (f,args) in model.heatmap_iterations.items() }

  # Calculate Heatmap data and flip Labels out, split data and args, add average=True arg
  data = [heatmap_data(model) for model in models]
  return  {label: {'data': [run[label][0] for run in data], 'args': (*data[0][label][1], True)} for label in data[0]}
