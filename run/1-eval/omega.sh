# Kappa=[32/8192] | Chi=0.5
# Envs: Maze9Sparse, PointMaze9 (Scale 4, reward, steps, hps, training steps)
P="results/1-eval/omega"; mkdir -p "$P"

# Point Maze
for RUN in 1 2 3 4 5 6 7 8; do
  for E in "Maze9Sparse 24" "PointMaze9 96"; do
    set -- $E; ENV=$1; T=$2;
    O="results/1-eval/out/omega/$ENV"; mkdir -p "$O"
    for OMEGA in "0.1" "0.5" "1" "2" "10"; do
      nohup python -m run DIRECT -e $ENV -t $T --omega $OMEGA -s $RUN --path $P &> "$O/$OMEGA-$RUN.out" &
      sleep 2
    done
  done
done
