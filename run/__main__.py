import argparse; import setuptools; import os; import random
from safety_env import factory; from util import TrainableAlgorithm
from direct import DIRECT; from baselines import DQN, PPO
parser = argparse.ArgumentParser()

# General Arguments
parser.add_argument('algorithm', type=str, help='The algorithm to use', choices=['DIRECT', 'A2C', 'DQN', 'HER', 'PPO'])
parser.add_argument( '--env', nargs='+', default=['DistributionalShift', 0, 4, 1], metavar="Environment",
  help='The name and spec and of the safety environments to train and test the agent. Usage: --env NAME, CONFIG, N_TRAIN, N_TEST')
parser.add_argument('-s', dest='seed', type=int, help='The random seed. If not specified a free seed [0;999] is randomly chosen')
parser.add_argument('--load', help='Whether to load a model or train a new one.', action='store_true')
parser.add_argument('--test', help='Run in test mode (dont write log files).', action='store_true')

# Training Arguments
parser.add_argument('-t', dest='timesteps', type=float, default=10e4, help='Number of timesteps to learn the model (eg 10e4)')
parser.add_argument('-ts', dest='maxsteps', type=float, help='Maximum timesteps to learn the model (eg 10e4), using early stopping')
parser.add_argument('--reward-threshold', type=float, help='Threshold for 100 episode mean return to stop training.')

# DIRECT Specific Arguments  
parser.add_argument('--chi', type=float, default=1.0, help='The mixture parameter determining the mixture of real and discriminative reward. (default: 1.0)')
parser.add_argument('--kappa', type=int, default=512, help='Number of trajectories to be stored in the direct buffer. (default: 512)')
parser.add_argument('--omega', type=float, default=1.0, help='The frequency to perform discriminator updates in relation to policy updates. (default: 1/1)')

# Policy Optimization Arguments
parser.add_argument('--n-steps', type=int, help='The length of rollouts to perform policy updates on')

# Get arguments 
args = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}
if args['algorithm'] != "DIRECT": [args.pop(key) for key in ['chi', 'kappa', 'omega']]
hp_suffix = f"/{args['chi']}_{args['omega']}_{args['kappa']}" if args['algorithm'] == "DIRECT" else ''

# Get path & seed, create model & envs
algorithm = eval(args.pop('algorithm')); model = None
env = dict(zip(['name', 'spec', 'n_train', 'n_test'], args.pop('env')))
base_path = lambda seed: f"results/{algorithm.__name__}/{env['name']}{hp_suffix}/{seed}/"
gen_seed = lambda s=random.randint(0, 999): s if not os.path.isdir(base_path(s)) else gen_seed()
seed = args.pop('seed', gen_seed()); path = base_path(seed); envs = factory(seed, **env)
if args.pop('test'): path = None

# Extract training parameters & merge model args
reward_threshold = envs['train'].unwrapped.get_attr('spec')[0].reward_threshold
stop_on_reward = args.pop('reward_threshold',reward_threshold) if any(arg in ['maxsteps', 'reward_threshold'] for arg in args) else None
timesteps = args.pop('maxsteps', args.pop('timesteps'))
print(f"Stopping training at threshold {stop_on_reward}") if stop_on_reward else print(f"Training for {timesteps} steps")

#Create, train & save model 
args = {'envs': envs, 'path': path, 'seed': seed, **args}
load = args.pop('load', False) #; print(f"{'load' if load else 'creat'}ing model  with args {args}")
model:TrainableAlgorithm = algorithm.load(**args) if load else algorithm(**args)
model.learn(total_timesteps=timesteps, stop_on_reward=stop_on_reward, reset_num_timesteps = not load)
if path: model.save()
