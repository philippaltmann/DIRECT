# DIRECT 
Discriminative Reward Co-Training

## Setup 

### Requirements
- python 3.10

### Installation 
```sh
$ pip install -r requirements.txt
```

## Training

Example for training DIRECT:
```python 
from direct import DIRECT 

envs = 'Maze9Sparse'; epochs = 24
model = DIRECT(envs=envs, seed=42)
model.learn(total_timesteps=epochs * 2048 * 4)
model.save()
```

## Running Experiments 

```sh
# Train DIRECT and baselines 
python -m run DIRECT -e Maze9Sparse -t 24 --path 'results/1-eval'
python -m run [DIRECT|GASIL|SIL|A2C|PPO] -e FetchReach -t 96 --path 'results/2-bench'

# Display help for command line arguments 
python -m run -h

# Run Evaluation Scripts:
./test/1-eval/kappa.sh
./test/1-eval/omega.sh
./test/1-eval/chi.sh

# Run Benchmark Scripts:
./test/2-bench/maze.sh
./test/2-bench/shift.sh
./test/2-bench/fetch.sh
```

## Plotting
```sh
########
# EVAL #
########

# Kappa
python -m plot results/1-eval/kappa -m Buffer --merge Training Momentum Scores 

# Omega
python -m plot results/1-eval/omega -m Discriminator --merge Training

# Chi
python -m plot results/1-eval/chi -m DIRECT --merge Training


#############
# Benchmark #
#############

# Maze
python -m plot results/2-bench -e Maze9Sparse -m Training

# HoleyGrid
python -m plot results/2-bench -e HoleyGrid -m Shift --merge Training

# Fetch
python -m plot results/2-bench -e FetchReach -m Training
```