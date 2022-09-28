import numpy as np; import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

import os; import itertools; from parse import parse; from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
from os import path

def triangle_heatmap(data, min=0, max=1):
  # TODO: automatic min max scaling 
  figure, ax = plt.subplots(figsize=(8,6))
  # f = plt.figure(figsize=fig_size)
  ct = lambda c,r: (c,r) # Helper points at c(ollumn), r(ow) for w(idth)
  ul = lambda c,r,w=0.5: (c-w, r-w); ur = lambda c,r,w=0.5: (c+w, r-w)
  dr = lambda c,r,w=0.5: (c+w, r+w); dl = lambda c,r,w=0.5: (c-w, r+w)
  
  # For positioning text
  uc = lambda c,r,w=0.3: (c, r-w); rc = lambda c,r,w=0.3: (c+w, r)
  dc = lambda c,r,w=0.3: (c, r+w); lc = lambda c,r,w=0.3: (c-w, r)

  # Indices of points to build triangle (up, right, down, left)
  triangles = [(0,1,2), (0,2,3), (0,3,4), (0,4,1)]
  labels = lambda c,r: np.array([uc(c,r),rc(c,r),dc(c,r),lc(c,r)])
  points = lambda c,r: np.array([ct(c,r),ul(c,r),ur(c,r),dr(c,r),dl(c,r)])
  square = lambda c,r: Triangulation(*points(c,r).T, triangles)
  # plot = lambda v,c,r: ax.tripcolor(square(c,r), v, cmap='Spectral',vmin=-51, vmax=49, ec='w') #vmin=0, vmax=1, RdYlGn
  # text = lambda v,c,r: [ax.text(x,y, f'{v:.0f}', color='k' if -10 < v < 10 else 'w', ha='center', va='center') for x,y,v in zip(*labels(c,r).T, v)]
  
  plot = lambda v,c,r: ax.tripcolor(square(c,r), v, cmap='Spectral', vmin=min, vmax=max, ec='w') #, RdYlGn
  center = min + (max - min) / 2; width = abs(min) + abs(max)
  lcolor = lambda v: 'k' if center - 0.3 * width < v < center + 0.3 * width else 'w'
  _txt = lambda v,c,r: ax.text(c,r, f'{v:.2f}', color=lcolor(v), ha='center', va='center')
  text = lambda v,c,r: [_txt(v,x,y) for x,y,v in zip(*labels(c,r).T, v)]

  # Add triangles & labels
  imgs = [plot(v,c,r) for r, row in enumerate(data) for c, v in enumerate(row) if not (None in v)]
  [text(v,c,r) for r, row in enumerate(data) for c, v in enumerate(row) if len(v) if not (None in v)]

  # Add legends and axes
  figure.colorbar(imgs[0], ax=ax); ax.invert_yaxis();
  ax.set_xticks(range(len(data[0]))); ax.set_yticks(range(len(data))); 
  ax.margins(x=0, y=0); ax.set_aspect('equal', 'box'); plt.tight_layout()
  return figure


def fetch_experiments(base='./results', alg=None, env=None, metrics=[], load_csv=False, dump_csv=False):
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
  else: experiments = [{'algorithm': a.name, 'path': a} for a in subdirs(base)]

  print(f"Scanning for {env if env else 'environments'} in {base}")  # Second layer: Environments
  if env: experiments = [{**exp, 'env': env, 'path': f'{exp["path"]}/{env}'} for exp in experiments]
  else: experiments = [{**exp, 'env': e.name, 'path': e} for exp in tqdm(experiments) for e in subdirs(exp['path'])]

  print(f"Scanning for hyperparameters in {base}")  # Third layer: Hyperparameters & number of runs
  hp = lambda name: dict(zip(['chi','omega','kappa'], parse('{:.1f}_{:.1f}_{:d}', name) or []))
  experiments = [{ **exp, 'path': e.path, **hp(e.name), 'runs': len(subdirs(e)) }
    for exp in tqdm(experiments) if os.path.isdir(exp['path']) for e in subdirs(exp['path'])
  ]

  print(f"Loading experiment logfiles from {metrics}")  # Fourth layer: tb files   
  csv = lambda run, name: path.isfile(f'{run.path}/{name}.csv')
  extract_args = {'columns': ['Time', 'Step', 'Data'], 'index': 'Step', 'exclude': ['Time']}
  extract_data = lambda run, key : pd.DataFrame.from_records(EA(run.path).Reload().Scalars(key), **extract_args)
  extract_csvs = lambda run, name: pd.read_csv(f'{run.path}/{name}.csv').set_index('Step')
  load_scalars = lambda run, name, key: extract_csvs(run, name) if csv(run, name) else extract_data(run, key)
  process_data = lambda exp, name, key: [ load_scalars(run, name, key) for run in tqdm(subdirs(exp['path'])) ] 

  # Process given experiments
  experiments = [{**exp, 'data': { name: process_data(exp, name, key) for name, key in metrics } } for exp in tqdm(experiments)]

  dump_experiment = lambda data, runs: [ df.to_csv(f'{r.path}/{m}.csv') for m, d in data for r, df in zip(runs, d) ]
  if dump_csv: [dump_experiment(exp['data'].items(), subdirs(exp['path'])) for exp in experiments]

  return experiments
