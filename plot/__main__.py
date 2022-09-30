"""
Example commands:
python -m vizualisation experiments/1-HP/sar -g algorithm env omega
python -m vizualisation experiments/1-HP/sar

TODO: consider smoothing?
"""
import argparse; import os; 
from common.metrics import *
from common.plotting import * 

# Process commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument('base', default='./results', help='The results root')
parser.add_argument('-a', dest='alg', help='Algorithm to vizualise.')
parser.add_argument('-e', dest='env', help='Environment to vizualise.')
parser.add_argument('-g', dest='groupby', nargs='+', default=['algorithm', 'env'], metavar="groupby", help='Experiment keys to group plotted data by.')
args = parser.parse_args()

metrics = [ # Title, Scalar(Name, Tag), process(scalar)->data, display(data)->trace
  ('Reward', ('Reward', 'metrics/validation_reward'), process_ci, plot_ci),
  ('Return', ('Return', 'rewards/return-100-mean'), process_ci, plot_ci),
  ('Buffer', ('Buffer', 'rewards/return-100-mean'), process_ci, plot_ci),
  ('Length', ('Length', 'rewards/length-100-mean'), process_ci, plot_ci),
  ('Steps', ('Reward', 'metrics/validation_reward'), process_steps, plot_box)]
titles, scalars, procs, plotters = zip(*metrics)

# Load, sort and group experiments, calculate metrics and generate figures
experiments = fetch_experiments(args.base, metrics=list(zip(titles, scalars)), dump_csv=True)
experiments = sorted(experiments, key=lambda exp: (exp['kappa'],exp['omega']))
experiments = group_experiments(experiments, args.groupby)
experiments = calculate_metrics(experiments, list(zip(titles, procs)))
figures = generate_figures(experiments, dict(zip(titles, plotters)))

# Save figures
out = f'{args.base}/plots/{"-".join(args.groupby)}'; os.makedirs(out, exist_ok=True)
[figure.write_image(f'{out}/{name}.pdf') for name, figure in figures.items()]