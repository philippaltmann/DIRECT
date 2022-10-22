# Config -> Name 
name_lookup = {'DistributionalShift': 'Dense', 'DistributionalShift-Sparse': 'Sparse'}
spec_lookup = {'0': 'Training', '1': 'ObstacleShift', '2': 'GoalShift'}
env_name = lambda name, spec: name_lookup[name] + spec_lookup[spec]
env_conf = lambda name: { sv+nv: (nk, sk)for nk,nv in name_lookup.items() for sk, sv in spec_lookup.items() }[name]
