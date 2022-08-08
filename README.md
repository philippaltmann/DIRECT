# DIRECT 
Discriminative Reward Co-Training

## Setup 

### Requirements
- python 3.8.6
- cuda 11.3 or 10.2

### Installation 
```sh
$ pip install -r requirements.txt
```

### Training

Using the python runner: 
```sh
# Run DIRECT Training for 8192 steps
$ python -m run DIRECT -t 8192

# ..., stopping early [at specified threshold]
$ python -m run DIRECT -ts 8192 --reward-threshold 

# Reload model with seed 42
$ python -m run DIRECT --load -s 42

# Display help for command line arguments 
$ python -m run -h
```

Example for training direct in safety environments:
```python 
from direct import DIRECT 
from util import TrainableAlgorithm
from safety_env import factory

envs = factory(seed=42, name='DistributionalShift')
model:TrainableAlgorithm = DIRECT(envs=envs, seed=42, path='results/DIRECT/42', chi=1.0, kappa=512, omega=1/1)

# Evaluation is done within the training loop
model.learn(total_timesteps=10e4, stop_on_reward=True)
model.save(base_path+"models/trained")

reload = algorithm.load(envs=envs, path=base_path+"models/trained")
reload.learn(total_timesteps=512, reset_num_timesteps=False)
```

## Fix Warnings in used packages 

`/.direnv/python-3.7.7/lib/python3.7/site-packages/pycolab/ascii_art.py:318: FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple. Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error in the future.`

change ascii_art.py line 318 from 
`art = np.vstack(np.fromstring(line, dtype=np.uint8) for line in art)`
to 
`art = np.vstack([np.fromstring(line, dtype=np.uint8) for line in art])`