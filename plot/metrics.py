import os; from os import path; import itertools; from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
import pandas as pd; import numpy as np; import scipy.stats as st; import re
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym; from hyphi_gym import named, Monitor
from stable_baselines3.common.monitor import Monitor as SBM
from scipy import nanmean
from baselines import *

HPS = {
  'χ': [.0, .25, .5, .75 ,1.], 
  'κ': [8, 32, 256, 2048, 8192], 
  'ω': [.1, .5, 1., 2., 10.]
}


def extract_model(exp, run):
  algorithm, seed = eval(exp['algorithm']), int(run.name)
  model = algorithm.load(load=run.path, seed=seed, envs=[exp['env']], path=None, device='cpu', silent=True)
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

  print(f"Scanning for {env if env else 'environments'} in {base}")  # First layer: Environments
  if env: experiments = [{'env': e.name, 'path': e.path} for e in subdirs(base) if env == e.name] # in
  else: experiments = [{'env': e.name, 'path': e.path} for e in subdirs(base) if e.name != 'plots']

  def extract_hp(name):
    hps = {'info': ' '.join(name.split(' ')[1:])}
    for hp,v in HPS.items(): 
      t = type(v[0])
      if hp in hps['info']: hps[hp] = t(re.search(r'\['+hp+':(.*?)\]', hps['info']).group(1))
    return hps

  print(f"Scanning for {alg if alg else 'algorithms'} in {base}")  # Second layer: Algorithms
  if alg: experiments = [{**exp, 'algorithm': alg, 'path': a} for exp in tqdm(experiments) for a in subdirs(exp['path']) if alg == a.name]
  else: experiments = [{**exp, 'algorithm': a.name.split(' ')[0], **extract_hp(a.name), 'path': a} for exp in tqdm(experiments) for a in subdirs(exp['path']) if any([A in a.name for A in ALGS])]

  # Third Layer: Count Runs / fetch tb files
  print(f"Scanning for hyperparameters in {base}")  # Third layer: Hyperparameters & number of runs
  experiments = [{ **e, 'seeds': [int(s.name) for s in subdirs(e['path'])] } for e in tqdm(experiments) if os.path.isdir(e['path'])] 

  progressbar = tqdm(total=sum([len(exp['seeds']) for exp in experiments])* len(metrics))
  data_buffer = {}

  def fetch_data(exp, run_path, name, key):
    # Load data from csv if possible
    if path.isfile(f'{run_path}/{name}.csv'): return pd.read_csv(f'{run_path}/{name}.csv').set_index('Step')
    if 'Model' in key: return f'{run_path}/{name}.csv' if dump_csv else None
    if 'Buffer' in name and 'DIRECT' not in run_path: return None
    if 'DIRECT' in name and 'DIRECT' not in run_path: return []

    # Use buffered Event Accumulator if already open
    if log := data_buffer.get(run_path):
      extract_args = {'columns': ['Time', 'Step', 'Data'], 'index': 'Step', 'exclude': ['Time']}
      data = pd.DataFrame.from_records([(s.wall_time, s.step, s.value) for s in log.Scalars(key)], **extract_args)
      data = data.loc[~data.index.duplicated(keep='first')] # Remove duplicate indexes
      if name == 'Training': data.iloc[0] = np.NaN  # Remove initial zero
      if dump_csv: data.to_csv(f'{run_path}/{name}.csv')
      return data
    
    data_buffer.update({run_path: EA(run_path + "/train/").Reload()})
    return fetch_data(exp, run_path, name, key)

  def extract_data(exp, run, name, key):
    progressbar.update()
    data = fetch_data(exp, run.path, name, key)
    if not isinstance(data, str) and len(data): data.index /= (4*2048) # Steps -> Rollouts 

    if 'Buffer' in name and exp['algorithm'] == 'DIRECT':
      if 'Buffers' in name:
        data = {b.name[:-4]: np.load(b.path) for b in os.scandir(f'{run.path}/buffer') if b.name != '0.npy'}
      else:
        final = 786432 if 'Point' in exp['env'] else 196608 # int(data.index[-1]*(4*2048))
        data = {str(final): np.load(f'{run.path}/buffer/{final}.npy')}

    if name in ['Scores', 'Momentum']:
      initial = -400 if 'Point' in exp['env'] else -100 if 'Maze' in exp['env'] else 0
      data = pd.concat([pd.DataFrame([initial], columns=['Data']), data]) #/ 2048; data.index.name = 'Step'

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


def group_experiments(experiments, groupby=['env']):
  # Graphical helpers for titles, labels
  title = lambda exp: ' '.join([exp[key] for key in ['algorithm', 'env'] if key in exp and key in groupby])
  options = list(itertools.product(*[ list(dict.fromkeys([exp[group] for exp in experiments])) for group in groupby ]))
  ingroup = lambda experiment, group: all([experiment[k] == v for k,v in zip(groupby, group)])
  options = list(zip(options, [[ title(exp) for exp in experiments if ingroup(exp,group)] for group in options ]))
  options = [(group, [0, len(titles)], titles[0]) for group, titles in options]

  # Set alg as label if muliple algs, else set hps
  if len(set([e['algorithm'] for e in experiments])) > 1: label = lambda exp: exp['algorithm']
  else: label = lambda exp: '-'.join([f"{k}:{exp[k]}" for k in HPS.keys() if k in exp])
  return [{'title': title, **dict(zip(groupby,group)), 'graphs': [{ 'label': label(exp), **exp} for exp in experiments if ingroup(exp, group) ] } for group, index, title in options ]


def calculate_metrics(plots, metrics):
  """Given a set of experiments and metrics, applies metric calulation
  :param plots: list of dicts containing plot information and graphs with raw data
  :param metrics: Dicts of metric names and calculation processes 
  """
  for plot in plots: 
    env = SBM(Monitor(gym.make(**named(plot['env']))))
    seeds = list(set([s for g in plot['graphs'] for s in g['seeds']]))
    reward_threshold, reward_range = env.get_wrapper_attr('reward_threshold'), env.unwrapped.reward_range
    if reward_threshold == 'VARY': 
      def iterget(key, seed): env.reset(seed=seed); return getattr(env,key)
      reward_threshold = np.array([iterget('reward_threshold', s) for _ in range(int(100/len(seeds))) for s in seeds]).mean()
      reward_range = tuple(np.array([iterget('reward_range', s) for _ in range(int(100/len(seeds))) for s in seeds]).mean(0))
    env.info = {'seeds': seeds, 'reward_range':reward_range, 'reward_threshold': reward_threshold}
    plot['env'] = env 
  return [ result for metric, process in metrics for plot in plots for result in process(metric, plot)]


def prepare_ci(data, bounds=None, scale=1):
  # Helper to claculate confidence interval
  ci = lambda d, confidence=0.95: st.t.ppf((1+confidence)/2, len(d)-1) * st.sem(d)
  # Prepare Data (fill until highest index)
  if len(data[0]) == 0: return ([],[])
  steps = [d.index[-1] for d in data]; maxsteps = np.max(steps)
  for d in data: d.at[maxsteps, 'Data'] = float(d.tail(1)['Data'].iloc[0])
  data = pd.concat(data, axis=1, ignore_index=False, sort=True).bfill()
  
  # Mean 1..n | CI 1..n..1
  mean, h = data.mean(axis=1), data.apply(ci, axis=1)
  ci = pd.concat([mean-h, (mean+h).iloc[::-1]])
  if bounds is not None: ci = ci.clip(*bounds[::-1])
  return (mean, ci)


def prepare_eval(env, model, callback=lambda g,l: None, **kwargs):
  kwargs = {'n_eval_episodes':100, 'deterministic':False, 'return_episode_rewards':True, **kwargs} 
  env.reset(seed=model.seed); # print("Running evaluation")
  results = evaluate_policy(model=model, env=env, callback=callback, **kwargs)
  return results


def prepare_histogram(env, buffer):
  """3D Histogram creation according to 
  https://chart-studio.plotly.com/%7Eempet/15255/plotly-3d-barchart-or-histogram3d-for-2d/#/"""
  # if env.unwrapped.step_scale > 1: _pos = lambda obs: obs[2:4] + np.array(env.unwrapped.size) / 2 - 0.5
  cont = 'Point' in env.unwrapped.name; #env.unwrapped.step_scale > 1
  if cont: _pos = lambda obs: tuple(int(o) for o in (obs[2:4] - np.array(env.unwrapped.size) / 2 - 0.5).round())
  else: _pos = lambda obs: tuple(env.unwrapped.getpos(board=obs.reshape(env.unwrapped.size)))
  x, y = [np.array(p) for p in zip(*[_pos(obs) for obs in buffer])]
  
  # x, y- array-like of shape (n,), defining the x, and y-ccordinates of data set for which we plot a 3d hist
  heatmap = np.zeros(env.unwrapped.size); gap = 0.1
  def update_heatmap(pos): heatmap[pos] += 1 
  [update_heatmap(p) for p in zip(x,y)]
  if cont: heatmap = np.rot90(heatmap)
  sizes = np.array([[1-gap, 1-gap, n] for row in heatmap for n in row])
  positions = np.array([[x,y,0] for x in range(env.unwrapped.size[0]) for y in range(env.unwrapped.size[1])])

  # Create bar vertices
  assert isinstance(sizes, np.ndarray) and isinstance(positions, np.ndarray) and sizes.shape == positions.shape
  bar = np.array([[0, 0, 0],[1, 0, 0],[1, 1, 0],[0, 1, 0],[0, 0, 1],[1, 0, 1],[1, 1, 1],[0, 1, 1]], dtype=float)
  bars = [bar*size+pos for pos, size in zip(positions, sizes) if size[2]!=0]; p, q, r = np.array(bars).shape 
  # extract unique vertices from the list of all bar vertices
  vertices, ixr = np.unique(np.array(bars).reshape(p*q, r), return_inverse=True, axis=0)
  X, Y, Z = vertices.T; I,J,K = [],[],[] 
  # for each bar, derive the sublists of indices i, j, k assocated to its chosen triangulation
  for k in range(len(bars)): #Perform triangualation
    I.extend(np.take(ixr, [8*k, 8*k+2,8*k, 8*k+5,8*k, 8*k+7, 8*k+5, 8*k+2, 8*k+3, 8*k+6, 8*k+7, 8*k+5])) 
    J.extend(np.take(ixr, [8*k+1, 8*k+3, 8*k+4, 8*k+1, 8*k+3, 8*k+4, 8*k+1, 8*k+6, 8*k+7, 8*k+2, 8*k+4, 8*k+6])) 
    K.extend(np.take(ixr, [8*k+2, 8*k, 8*k+5, 8*k, 8*k+7, 8*k, 8*k+2, 8*k+5, 8*k+6, 8*k+3, 8*k+5, 8*k+7]))  
  
  return {'x':X, 'y':Y, 'z':Z, 'i':I, 'j':J, 'k':K}


def prepare_heatmap(env, models, key=None):
  if key is None: s = env.unwrapped.sparse; env.unwrapped.sparse = False
  def iterate(env, function, fallback=np.nan):
    """Iterate all possible actions in all env states, apply `funcion(env, state, action)`
    function: `f(env, state, action, reward()) => value` to be applied to all actions in all states 
      default: return envreward
    fallback: result to put in fields where agent can't be moved (eg. walls)
    :returns: ENVxACTIONS shaped function results"""

    # Get env object, prepare mappings for env generation 
    from hyphi_gym.envs.common.board import CELLS, WALL, FIELD, AGENT, TARGET, ACTIONS
    empty_board = np.reshape(env.reset()[0], env.unwrapped.size);  fallback = [fallback] * len(ACTIONS)
    
    # Create empty board for iteration & function for reverting Observation to board 
    empty_board[empty_board == CELLS[AGENT]] = CELLS[FIELD]
    reward = lambda action: lambda: env.step(action)[1]
    def prepare(x,y): state = empty_board.copy(); state[y][x] = CELLS[AGENT]; return state
    norm = lambda x: abs(x)/np.sum(abs(x)) * 100 if np.sum(abs(x)) != 0 else x # Norm to %
    process = lambda state: norm(np.array([function(env, env.reset(layout=state)[0], action, reward(action)) if action is not None else np.nan for action in ACTIONS]))
    return np.array([ 
      [ process(prepare(x,y)) if cell == CELLS[FIELD] else fallback for x, cell in enumerate(row) ] 
        for y, row in enumerate(empty_board)
    ])
  def fallback(e,s,a,r): return r()
  heatmap = nanmean([iterate(env, model.heatmap_iterations[key][0] if key is not None else fallback) for model in models], axis=0)
  if key is None: env.unwrapped.sparse = s
  return heatmap


def process_metric(metric, plot):
  return [{ **plot, 'graphs':  [{ 
    **graph, 'data': prepare_ci(graph['data'][metric]) 
  } for graph in plot['graphs']], 'metric': metric}]

def process_return(metric, plot):
  return [{ **plot, 'graphs':  [{ 
    **graph, 'data': prepare_ci(graph['data'][metric], bounds=plot['env'].info['reward_range']) 
  } for graph in plot['graphs']], 'metric': metric}]


def process_buffer(metric, plot): 
  steps = lambda graph: set(s for b in graph['data'][metric] for s in b.keys())
  buffer = lambda graph, step: [s for b in graph['data'][metric] for s in b[step]]
  return [{
    **{ key: { **plot, 'graphs': [{ **graph, 'data': prepare_ci(graph['data'][key]) } 
        for graph in plot['graphs']], 'metric': key}
      for key in plot['graphs'][0]['data'].keys() if key != metric },
    'heatmap': { **plot, 'graphs': [ 
      { **graph, 'data': [ prepare_histogram(plot['env'], buffer(graph, step)) for step in steps(graph)]}
      for graph in plot['graphs']
    ], 'metric': metric},
    'metric': metric, 'env': plot['env']
  }]


def process_direct(metric, plot): 
  return [{
    'direct':{ **plot, 'graphs': [{ 
      **graph, 'data': prepare_ci(graph['data'][metric])#, bounds=plot['env'].info['reward_range'] 
      # **graph, 'data': prepare_ci(data(graph)[0], scale=graph['χ'])  #, scale=graph['κ']
    } for graph in plot['graphs']], 'metric': metric},
    'heatmap': { **plot, 'graphs': [ 
      { **graph, 'data': prepare_heatmap(plot['env'], graph['models'], 'direct' if graph['χ'] > 0 else None) }
      for graph in plot['graphs'] if plot['env'].get_wrapper_attr('discrete')
    ], 'metric': metric},
    'metric': metric, 'env': plot['env']
  }]


def process_shift(metric, plot): 
  plot['env'].unwrapped.shift()
  return [{
    'shift':{ **plot, 'graphs': [{ 
      **graph, 'data': prepare_ci(graph['data'][metric])
    } for graph in plot['graphs']], 'metric': metric},
    'heatmap': { **plot, 'graphs': [ 
      { **graph, 'data': prepare_heatmap(plot['env'], graph['models'], 'policy')} #action
      for graph in plot['graphs'] if plot['env'].get_wrapper_attr('discrete')
    ], 'metric': metric},
    'metric': metric, 'env': plot['env']
  }]


def process_eval(metric, plot):
  def _eval(data, model, progress):
    if isinstance(data, pd.DataFrame): return data
    R = pd.DataFrame(prepare_eval(plot['env'], model)[0], columns=['Return']); R.index.name = 'Step'
    if data is not None: R.to_csv(data)
    progress.update(1)
    return R
  print("Running evaluations")
  with tqdm(total=len(plot['graphs'])*len(plot['graphs'][0]['models'])) as progress:
    return [{ **plot, 'graphs':  [{ 
      **graph, 'data': pd.concat([_eval(data, model, progress) for data, model in zip(graph['data'][metric], graph['models'])])['Return']
    } for graph in plot['graphs']], 'metric': metric}]
