"""
Example commands:
python -m plot experiments/1-HP/sar -g algorithm env omega
python -m plot experiments/2-benchmark -g env
python -m plot experiments/2-benchmark -g env -a PPO -m Heatmap
python -m plot experiments/1-HP/sar

TODO: consider smoothing?
"""
import argparse; import os; 
from plot.metrics import *
from common.plotting import * 

options = { # Title, Scalar(Name, Tag), process(scalar)->data, display(data)->trace
  'Reward': (('Reward', 'metrics/validation_reward'), process_ci, plot_ci),
  'Return': (('Return', 'rewards/return-100-mean'), process_ci, plot_ci),
  'Buffer': (('Buffer', 'rewards/return-100-mean'), process_ci, plot_ci),
  'Length':  (('Length', 'rewards/length-100-mean'), process_ci, plot_ci),
  'Steps': (('Reward', 'metrics/validation_reward'),  process_steps, plot_box),
  'Heatmap': (('Model', 0), process_heatmap, plot_heatmap)
}

# Process commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument('base', default='./results', help='The results root')
parser.add_argument('-a', dest='alg', help='Algorithm to vizualise.')
parser.add_argument('-e', dest='env', help='Environment to vizualise.')
parser.add_argument('-g', dest='groupby', nargs='+', default=['algorithm', 'env'], metavar="groupby", help='Experiment keys to group plotted data by.')
parser.add_argument('-m', dest='metrics', nargs='+', default=['Return', 'Steps'], choices=options.keys(), help='Experiment keys to group plotted data by.')
parser.add_argument('-nd', dest='dump_csv', action='store_false', help='Skip csv dump')
args = vars(parser.parse_args()); groupby = args.pop('groupby'); 

metrics = [(metric, *options[metric]) for metric in args.pop('metrics')]
titles, scalars, procs, plotters = zip(*metrics)

# Load, sort and group experiments, calculate metrics and generate figures
experiments = fetch_experiments(**args, metrics=list(zip(titles, scalars)))
experiments = group_experiments(experiments, groupby)
experiments = calculate_metrics(experiments, list(zip(titles, procs)))
figures = generate_figures(experiments, dict(zip(titles, plotters)))

# Save figures
out = f'{args["base"]}/plots/{"-".join(groupby)}'; os.makedirs(out, exist_ok=True)
[figure.write_image(f'{out}/{name}.pdf') for name, figure in figures.items()]