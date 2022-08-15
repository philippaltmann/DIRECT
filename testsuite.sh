# python -m run DIRECT -ts 10e5 --chi 1.0 --kappa 512  --omega 1/1

# Omega: 0.25 0.5 1.0 2 4
# chi 0 0.25 0.5 0.75 1.0
# Kappa 256 512 1024 2048 

for chi in 0.0 0.25 0.5 0.75 1.0; do
  for omega in 0.25 0.5 1.0 2.0 4.0; do
    for kappa in 256 512 1024 2048; do
      for run in 1 2 3 4; do
        nohup python -m run DIRECT -ts 10e5 --chi $chi --kappa $kappa --omega $omega &
        sleep 1
      done
    done
  done
done

for run in 1 2 3 4; do
  nohup python -m run PPO -ts 10e5 --chi $chi --kappa $kappa --omega $omega &
  sleep 1
done