import tensorboard

from direct import DIRECT
from safety_env import factory
from util import TrainableAlgorithm, PPO

algorithm = DIRECT #PPO
chi, kappa, omega = 1.0, 256, 1/1
stop_on_reward = True 
seed = 42 # random.randint(0, 999)

load = False # if !load -> generate seed, else load seed

# env_name, env_spec = "BoatRace", 0 # Index of env config to be used (cf. safety_env/__init__.py)
env_name, env_spec = "DistributionalShift", 0 # Index of env config to be used (cf. safety_env/__init__.py)
STAGES = { "train": { "n_envs": 4, "seed": seed }, "test": {"n_envs": 1, "seed": seed } } 
envs = {stage: factory(env_name, stage, args, env_spec) for stage, args in STAGES.items()}

base_path = "results/{}/{}/{}/"
# while os.path.isdir(base_path.format(algorithm.__name__, env_name, seed)): seed = random.randint(0, 999)
base_path = base_path.format(algorithm.__name__, env_name, seed)

if load: model:TrainableAlgorithm = algorithm.load(base_path+"models/trained", env=envs['train'])
# else: model:TrainableAlgorithm = algorithm(chi=chi, kappa=kappa, omega=omega, disc_kwargs={}, envs=envs, seed=seed, tb_path=base_path)
else: model:TrainableAlgorithm = algorithm(envs=envs, seed=seed, tb_path=base_path, omega=1/1, n_steps=64)

model.learn(total_timesteps=10e4, stop_on_reward=stop_on_reward) #10e4#8192#4096#10e5#
model.save(base_path+"models/trained")

