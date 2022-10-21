import argparse; import numpy as np; import plotly.graph_objects as go; import gym
from safety_env import *#factory, SAFETY_ENVS, heatmap_2D, heatmap_3D, env_id, en

env_names = [''.join([env,mode]) for env in SAFETY_ENVS.keys() for mode in ['', '-Sparse']]

parser = argparse.ArgumentParser()
parser.add_argument('env_name', type=str, help='The env to use', choices=env_names)
parser.add_argument('--play', type=str, help='Run environment in commandline.') #, action='store_true'
parser.add_argument('--plot', nargs='+', default=[], help='Save heatmap vizualisation plots.')

parser.add_argument('--seed', default=42, type=int, help='The random seed.')

args = parser.parse_args()
if args.play is not None: 
  if args.env_name.endswith('-Sparse'): assert False, 'Playing sparse envs is not supported'
  spec = int(args.play) if args.play.isdecimal() else args.play
  env = gym.make(env_id(args.env_name, spec))
  env.play()
specs = [int(s) if s.isdigit() else s for s in args.plot]
if len(specs):
  name, sparse = (args.env_name[:-7], True) if args.env_name.endswith('-Sparse') else (args.env_name, False)
  envs = { env_id(args.env_name, spec): make(name, spec, seed=args.seed, wrapper_kwargs={ "sparse": sparse }) for spec in specs }
  reward_data = { key: np.expand_dims(np.array(env.envs[0].iterate()), axis=3) for key, env in envs.items() }
  [heatmap_3D(data, -51,49, show_agent=True).write_image(f'results/plots/{key}-3D.pdf') for key, data in reward_data.items()]

  # reward_data = { f"{args.env_name}_{tag}": env.envs[0].iterate() for tag, env in envs['test'].items() }
  # [heatmap_2D(data, -51,49).savefig(f'results/plots/{key}.png') for key, data in reward_data.items()]
