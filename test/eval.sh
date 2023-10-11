for RUN in 1 2 3 4 5 6 7 8; do
  for SIZE in 9 15; do
    for ENV in "Maze${SIZE}Target";do
      for METHOD in "PPO" "DIRECT" "VPG" "SAC" "DQN"; do 
        O="results/2-eval/out/$ENV/$METHOD"; mkdir -p "$O"
        nohup echo "Running $METHOD $ENV $RUN"
        nohup python -m run $METHOD -t 196608 -e $ENV -s $RUN --path results/2-eval &> "$O/$RUN.out" &
        sleep 5
      done 
    done
  done
done
