import argparse; import numpy as np; import plotly.graph_objects as go; import gym
from safety_env import factory, SAFETY_ENVS, heatmap_2D, heatmap_3D, env_id

env_names = [''.join([env,mode]) for env in SAFETY_ENVS.keys() for mode in ['', '-Sparse']]

parser = argparse.ArgumentParser()
parser.add_argument('env_name', type=str, help='The env to use', choices=env_names)
parser.add_argument('--play', type=str, help='Run environment in commandline.') #, action='store_true'
parser.add_argument('--plot', help='Save heatmap vizualisation plots.', action='store_true')
parser.add_argument('--seed', default=42, type=int, help='The random seed.')

args = parser.parse_args()
if args.play is not None: 
  if args.env_name.endswith('-Sparse'): assert False, 'Playing sparse envs is not supported'
  spec = int(args.play) if args.play.isdecimal() else args.play
  env = gym.make(env_id(args.env_name, spec))
  env.play()

if args.plot:
  envs = factory(seed=args.seed, name=args.env_name)
  reward_data = { f"{args.env_name}_{tag}": np.expand_dims(np.array(env.envs[0].iterate()), axis=3) for tag, env in envs['test'].items() }
  [heatmap_3D(data, -51,49).write_image(f'results/plots/{key}-3D.pdf') for key, data in reward_data.items()]

  # reward_data = { f"{args.env_name}_{tag}": env.envs[0].iterate() for tag, env in envs['test'].items() }
  # [heatmap_2D(data, -51,49).savefig(f'results/plots/{key}.png') for key, data in reward_data.items()]
