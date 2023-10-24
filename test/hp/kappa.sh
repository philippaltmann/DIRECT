# Chi=0.5 | Omega=1.0
# Envs: Maze9Sparse, PointMaze9 (Scale 4, reward, steps, hps, training steps)
P="results/1-HP/kappa"; mkdir -p "$P"

# Point Maze
for RUN in 5 6 7 8; do #1 2 3 4
  for E in "Maze9Sparse 1" "PointMaze9 4"; do
    set -- $E; ENV=$1; F=$2;
    O="results/1-HP/out/kappa/$ENV"; mkdir -p "$O"
    for KAPPA in 8 32 256 2048 8192; do
      K=$((KAPPA*F)); T=$((F*24));
      nohup python -m run DIRECT --kappa $K -e $ENV -t $T -s $RUN --path $P &> "$O/$K-$RUN.out" &
      sleep 5
    done
  done
done