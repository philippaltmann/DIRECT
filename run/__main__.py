import argparse; import time
from algorithm import TrainableAlgorithm
from baselines import *
start = time.time()

# General Arguments
parser = argparse.ArgumentParser()
parser.add_argument('method', help='The algorithm to use')
parser.add_argument( '-e', dest='envs', nargs='+', default=['Maze7Target'], metavar="Environment", help='The name and spec and of the safety environments to train and test the agent. Usage: --env NAME, CONFIG, N_TRAIN, N_TEST')
parser.add_argument('-s', dest='seed', type=int, help='The random seed. If not specified a free seed [0;999] is randomly chosen')
parser.add_argument('-t', dest='timesteps', type=int, help='The number of training timesteps.', default=16*(2048*4)) #~10e5 128
parser.add_argument('--load', type=str, help='Path to load the model.')
# parser.add_argument('--load', type=str, help='Path to load the model.')
parser.add_argument('--test', help='Run in test mode (dont write log files).', action='store_true')
parser.add_argument('--stop', dest='stop_on_reward', help='Stop at reward threshold.', action='store_true') # TODO: test
parser.add_argument('--path', default='results', help='The base path, defaults to `results`')
parser.add_argument('-d',  dest='device', default='cpu') #, default='cuda', choices=['cuda','cpu', 'mps']

# Get arguments & extract training parameters & merge model args
args = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}; 
if args.pop('test'): args['path'] = None
timesteps = args.pop('timesteps')

if (load := args.pop('load', None)) is not None: 
  assert 'seed' in args, 'Seed to load required'
  mdir = f"/{args['envs'][0]}/{args['method']}/{args['seed']}"
  path = load+mdir; args['path'] += mdir

# Init Training Model
trainer = eval(args.pop('method'))
if load is not None: model = trainer.load(path, **args)
else: model:TrainableAlgorithm = trainer(**args)
  
print(f"Training {trainer.__name__ } in {args['envs'][0]} for {timesteps:.0f} steps.") 
model.learn(total_timesteps=timesteps)
if model.path: model.save()
print(f"Done in {time.time()-start}")
