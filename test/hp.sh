# Omega: 0.25 0.5 1.0 2 4
# Chi 0 0.25 0.5 0.75 1.0
# Kappa 256 512 1024 2048 

ENV="DistributionalShift-Sparse"
BASE="experiments/1-HP/full"
OUT="experiments/logfiles/direct"

for run in 1 2 3 4 5 6 7; do
  for chi in 1.0; do #0.75 0.5 0.25 0.0
    for omega in 1.0 2.0 4.0 0.5 0.25; do
      for kappa in 256 512 1024 2048 4096; do
        echo "Running DIRECT $chi-$omega-$kappa"
        python -m run DIRECT -ts 10e5 --chi $chi --kappa $kappa --omega $omega --env $ENV --path $BASE &> "$OUT/DIRECT_$chi-$omega-$kappa_$(date +%s).out" &
        sleep 1
      done
      sleep 1h
    done
  done
done
