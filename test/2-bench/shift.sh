# Benchmarking the exploration of shifting environments 
P="results/2-bench"; mkdir -p "$P"; O="$P/out/HoleyGrid"; mkdir -p "$O"

for RUN in 1 2 3 4 5 6 7 8; do
  for ALG in 'DIRECT' 'GASIL' 'SIL' 'PPO' 'A2C' 'PrefPPO' 'VIME'; do 
    nohup python -m run $ALG -e HoleyGrid HoleyGridShift -t 8 -s $RUN --path $P &> "$O/$ALG-$RUN.out" &
    sleep 2
  done
done
