# Chi=0.5 | Omega=1.0
# Envs: Maze9Sparse, PointMaze9 (Scale 4, reward, steps, hps, training steps)
P="results/1-HP/kappa"; mkdir -p "$P"

# Grid Maze
for RUN in 1 2 3 4; do
  for ENV in "Maze9Sparse"; do
    O="results/1-HP/out/kappa/$ENV"; mkdir -p "$O"
    for KAPPA in 256 512 1024 2048 4096; do
      nohup python -m run DIRECT --kappa $KAPPA -e $ENV -t 24 -s $RUN --path $P &> "$O/$KAPPA-$RUN.out" &
      sleep 5
    done
  done
done

# Point Maze
for RUN in 1 2 3 4; do
  for ENV in "PointMaze9"; do
    O="results/1-HP/out/kappa/$ENV"; mkdir -p "$O"
    for KAPPA in 32 128 1024 8192 32768; do
      nohup python -m run DIRECT --kappa $KAPPA -e $ENV -t 96 -s $RUN --path $P &> "$O/$KAPPA-$RUN.out" &
      sleep 5
    done
  done
done
