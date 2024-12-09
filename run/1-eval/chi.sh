# Kappa=[32/8192], Omega=1
# Envs: Maze9Sparse, PointMaze9 (Scale 4, reward, steps, hps, training steps)
P="results/1-eval/chi"; mkdir -p "$P"

# Point Maze
for RUN in 1 2 3 4 5 6 7 8; do
  for E in "Maze9Sparse 24" "PointMaze9 96" ; do
    set -- $E; ENV=$1; T=$2;
    O="results/1-eval/out/chi/$ENV"; mkdir -p "$O"
    for CHI in "0" "0.25" "0.5" "0.75" "1"; do
      nohup python -m run DIRECT -e $ENV -t $T --chi $CHI -s $RUN --path $P &> "$O/$CHI-$RUN.out" &
      sleep 2
    done
  done
done
