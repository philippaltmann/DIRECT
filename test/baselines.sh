ENV="DistributionalShift-Sparse"
BASE="experiments/2-benchmark"
OUT="experiments/logfiles/baselines"

for run in 1 2 3 4 5 6 7 8; do
  for alg in A2C DQN PPO VPG; do
    echo "Running $alg $run"
    nohup python -m run $alg -ts 10e5 --env $ENV --path $BASE &> "$OUT/$alg-$run.out" &
    sleep 1
  done
  sleep 20m
done

