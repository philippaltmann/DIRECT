for RUN in 1 2 3 4 5 6 7 8; do
  for SIZE in 11 13; do
    for METHOD in "PPO" "DIRECT" "VPG" "SAC" "DQN"; do 
      O="results/2-bench/out/$ENV/$METHOD"; mkdir -p "$O"
      ENV="PointMaze".$SIZE."Target -t 786432"
      nohup echo "Running $METHOD $ENV $RUN"
      nohup python -m run $METHOD -e $ENV -s $RUN --path results/2-bench &> "$O/$RUN.out" &
      sleep 1
    done 
  done
done
