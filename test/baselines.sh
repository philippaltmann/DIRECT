
ENV="DistributionalShift-Sparse"
BASE="experiments/2-benchmark"
OUT="experiments/logfiles/baselines"

# ENV="DistributionalShift"
# BASE="experiments/2-benchmark-dense"
# OUT="experiments/logfiles/baselines-dense"
sleep 6h

for RUN in 1 2 3 4 5 6 7 8; do # 1
  for ALG in A2C DQN ; do #PPO VPG
    echo "Running $ALG $RUN"
    nohup python -m run $ALG -ts 10e5 --env $ENV --path $BASE &> "$OUT/$ALG-$RUN.out" &
    # sleep 30m
  done
  sleep 30m
done
