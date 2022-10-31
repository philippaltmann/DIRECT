import setuptools; import argparse;
from safety_env import factory, env_name
from common import TrainableAlgorithm
from algorithm import *

# General Arguments
parser = argparse.ArgumentParser()
parser.add_argument('algorithm', type=str, help='The algorithm to use', choices=ALGS)
parser.add_argument( '--env', nargs='+', default=[], metavar="Environment", help='The name and spec and of the safety environments to train and test the agent. Usage: --env NAME, CONFIG, N_TRAIN, N_TEST')
parser.add_argument('-s', dest='seed', type=int, help='The random seed. If not specified a free seed [0;999] is randomly chosen')
parser.add_argument('--load', type=str, help='Path to load the model.')
parser.add_argument('--test', help='Run in test mode (dont write log files).', action='store_true')
parser.add_argument('--path', default='results', help='The base path, defaults to `results`')

# Training Arguments
parser.add_argument('-t', dest='timesteps', type=float, help='Number of timesteps to learn the model (eg 10e4)')
parser.add_argument('-ts', dest='maxsteps', type=float, default=10e5, help='Maximum timesteps to learn the model (eg 10e4), using early stopping')
parser.add_argument('--reward-threshold', type=float, help='Threshold for 100 episode mean return to stop training.')

# DIRECT Specific Arguments  
parser.add_argument('--chi', type=float, help='The mixture parameter determining the mixture of real and discriminative reward.')
parser.add_argument('--kappa', type=int, help='Number of trajectories to be stored in the direct buffer.')
parser.add_argument('--omega', type=float, help='The frequency to perform discriminator updates in relation to policy updates.')

# Policy Optimization Arguments
parser.add_argument('--n-steps', type=int, help='The length of rollouts to perform policy updates on')

# Get arguments 
args = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}; _default = lambda d,c: c + d[len(c):]

# Generate Envs and Algorithm
env = dict(zip(['name', 'spec', 'n_train', 'n_test'], _default(['DistributionalShift','0','4','1'], args.pop('env'))))
algorithm = eval(args.pop('algorithm')); model = None;  envs, sparse = factory(**env)
path = None if args.pop('test') else args.pop('path') + '/' + env_name(*list(env.values())[:2])

# Extract training parameters & merge model args
reward_threshold = envs['train'].get_attr('reward_threshold')[0] 
stop_on_reward = args.pop('reward_threshold',reward_threshold) if any(arg in ['maxsteps', 'reward_threshold'] for arg in args) else None
timesteps = args.pop('timesteps', args.pop('maxsteps'))
print(f"Stopping training at threshold {stop_on_reward}") if stop_on_reward else print(f"Training for {timesteps} steps")

#Create, train & save model 
args = {'envs': envs, 'path': path, **args} # 'seed': seed,
load = args.pop('load', False); #load = base_path(seed,load) if load else False
model:TrainableAlgorithm = algorithm.load(load=load, **args) if load else algorithm(**args) #device='cpu',
envs['train'].seed(model.seed); [env.seed(model.seed) for _,env in envs['test'].items()]
model.learn(total_timesteps=timesteps, stop_on_reward=stop_on_reward, reset_num_timesteps = not load)
if path: model.save()
