# Discriminative Reward Co-Training

[![DOI:10.1007/s00521-024-10512-8](https://zenodo.org/badge/doi/10.1007/s00521-024-10512-8.svg)](https://doi.org/10.1007/s00521-024-10512-8)
[![PDF](https://img.shields.io/badge/PDF-red.svg?labelColor=grey&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz48c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDU5NS4yOCA4NDEuODkiPgogIDxwYXRoICBzdHlsZT0iZmlsbDogd2hpdGUiIGQ9Ik01OTUuMjgsMjEwLjQ2aDB2LTMyTDQxNi44MSwwSDkwLjcxQzQwLjYxLDAsMCw0MC42MSwwLDkwLjcxdjY2MC40N2MwLDUwLjEsNDAuNjEsOTAuNzEsOTAuNzEsOTAuNzFoNDEzLjg2YzUwLjEsMCw5MC43MS00MC42MSw5MC43MS05MC43MVYyMTAuNDZoMFpNNTUwLjAyLDE3OC40NmgtNzQuNWMtMzIuMzcsMC01OC43MS0yNi4zNC01OC43MS01OC43MVY0NS4yNWwxMzMuMjEsMTMzLjIxWk01MDQuNTcsODA5Ljg5SDkwLjcxYy0zMi4zNywwLTU4LjcxLTI2LjM0LTU4LjcxLTU4LjcxVjkwLjcxYzAtMzIuMzcsMjYuMzQtNTguNzEsNTguNzEtNTguNzFoMjk0LjExdjg3Ljc1YzAsNTAuMSw0MC42MSw5MC43MSw5MC43MSw5MC43MWg4Ny43NXMwLDU0MC43MiwwLDU0MC43MmMwLDMyLjM3LTI2LjM0LDU4LjcxLTU4LjcxLDU4LjcxWiIvPgo8L3N2Zz4=)](https://rdcu.be/d3gj5)

This repository contains the implementation of Discriminative Reward Co-Training (DIRECT), a novel reinforcement learning extension designed to enhance policy optimization in challenging environments with sparse rewards, hard exploration tasks, and dynamic conditions. DIRECT integrates a self-imitation buffer for storing high-return trajectories and a discriminator to evaluate policy-generated actions against these stored experiences. By using the discriminator as a surrogate reward signal, DIRECT enables efficient navigation of the reward landscape, outperforming existing state-of-the-art methods in various benchmark scenarios. This implementation supports reproducibility and further exploration of DIRECT's capabilities.

![DIRECT Architecture](./assets/DIRECT.png "DIRECT Architecture")
![Evaluation Results](./assets/results.png "Evaluation Results")

## Setup

### Requirements

- python 3.10
- [hyphi_gym](https://pypi.org/project/hyphi-gym/)
- [Stable Baselines 3](https://pypi.org/project/stable-baselines3/)

### Installation

```sh
pip install -r requirements.txt
```

## Training

Example for training DIRECT:

```python
from baselines import DIRECT 

envs = ['Maze9Sparse']; epochs = 24
model = DIRECT(envs=envs, seed=42, path='results')
model.learn(total_timesteps = epochs * 2048 * 4)
model.save()
```

## Running Experiments

### Train DIRECT and baselines

```sh
python -m run DIRECT -e Maze9Sparse -t 24 --path 'results/1-eval'
python -m run [DIRECT|GASIL|SIL|A2C|PPO|VIME|PrefPPO] -e FetchReach -t 96 --path 'results/2-bench'
```

### Display help for command line arguments

```sh
python -m run -h
```

### Run Evaluation Scripts

```sh
./run/1-eval/kappa.sh
./run/1-eval/omega.sh
./run/1-eval/chi.sh
```

### Run Benchmark Scripts

```sh
./run/2-bench/maze.sh
./run/2-bench/shift.sh
./run/2-bench/fetch.sh
```

## Plotting

### Evaluation

#### Kappa

```sh
python -m plot results/1-eval/kappa -m Buffer --merge Training Momentum Scores 
```

#### Omega

```sh
python -m plot results/1-eval/omega -m Discriminator --merge Training
```

#### Chi

```sh
python -m plot results/1-eval/chi -m DIRECT --merge Training
```

### Benchmarks

#### Maze

```sh
python -m plot results/2-bench -e Maze9Sparse -m Training
```

#### HoleyGrid

```sh
python -m plot results/2-bench -e HoleyGrid -m Shift --merge Training
```

#### Fetch

```sh
python -m plot results/2-bench -e FetchReach -m Training
```
