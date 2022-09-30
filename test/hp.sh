#!/bin/sh
# Omega: 0.25 0.5 1.0 2 4
# Chi 0 0.25 0.5 0.75 1.0
# Kappa 256 512 1024 2048 

ENV="DistributionalShift-Sparse"
BASE="experiments/1-HP/sar"
OUT="experiments/logfiles/direct-sar"

for run in 1 2 3 4 5 6 7; do
  for chi in 1.0; do #0.75 0.5 0.25  #0;
    for omega in 1.0 2.0 4.0 0.5 0.25; do
      for kappa in 256 512 1024 2048; do
          nohup python -m run DIRECT -ts 10e5 --chi $chi --kappa $kappa --omega $omega --env $ENV --path $BASE &> "$OUT/DIRECT_$chi-$omega-$kappa_$(date +%s).out" &
          sleep 1
      done
      sleep 20m
    done
    sleep 1h
  done
done

# A2C
# for run in 1 2 3 4; do
#   nohup python -m run A2C -ts 10e5 --env $ENV &> "experiments/baselines/A2C-$run.out" &
#   sleep 1
# done

# DQN
# for run in 1 2 3 4; do
#   nohup python -m run DQN -ts 10e5 --env $ENV &> "experiments/baselines/DQN-$run.out" &
#   sleep 1
# done

# # PPO
# for run in 1 2 3 4; do
#   nohup python -m run PPO -ts 10e5 --env $ENV &> "experiments/baselines/PPO-$run.out" &
#   sleep 1
# done

# VPG
# for run in 1 2 3 4; do
#   nohup python -m run VPG -ts 10e5 --env $ENV &> "experiments/baselines/VPG-$run.out" &
#   sleep 1
# done