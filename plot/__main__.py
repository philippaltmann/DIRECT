"""Example commands:
Test evaluation
python -m plot plot/results/train -g env --eval 0 1 3
python -m plot plot/results/train --mergeon algorithm --eval 0 1 3

Generate Training plots:
python -m plot experiments/1-Train -m Return Steps -g env 
python -m plot experiments/1-Train -g env -a DIRECT --heatmap 0 1 3
python -m plot experiments/1-Train --mergeon algorithm --eval 0 1 3

Generate Adaptation plots:
python -m plot experiments/2-Adapt -b experiments/1-Train -m Return Steps -g env 
python -m plot experiments/2-Adapt -g env -a DIRECT --heatmap 0 1 3
python -m plot experiments/2-Adapt --mergeon algorithm --eval 0 1 3
python -m plot experiments/2-Adapt -g env --eval 0 1 3
"""

import argparse; import time; import os; 
from plot.metrics import *
from plot.plotting import * 

options = { # Title, Scalar(Name, Tag), process(scalar)->data, display(data)->trace
  # 'Reward': (('Reward', 'metrics/validation_reward'), process_ci, plot_ci),
  'Return': (('Return', 'rewards/return-100-mean'), process_ci, plot_ci),
  'Buffer': (('Buffer', 'rewards/return-100-mean'), process_ci, plot_ci),
  'Length':  (('Length', 'rewards/length-100-mean'), process_ci, plot_ci),
  'Steps': (('Return', 'rewards/return-100-mean'),  process_steps, plot_box), # ('Model', 'metrics/validation_reward')
}

# Process commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument('base', help='The results root')
parser.add_argument('-a', dest='alg', help='Algorithm to vizualise.')
parser.add_argument('-b', dest='baseline', help='Base path of reloaded model.')
parser.add_argument('-e', dest='env', help='Environment to vizualise.')
parser.add_argument('-g', dest='groupby', nargs='+', default=[], metavar="groupby", help='Experiment keys to group plotted data by.')
parser.add_argument('-m', dest='metrics', nargs='+', default=[], choices=options.keys(), help='Experiment keys to group plotted data by.')
parser.add_argument('--heatmap', nargs='+', default=[], help='Environment to vizualise.')
parser.add_argument('--mergeon', help='Key to merge experiments e.g. algorithm.')
parser.add_argument('--no-dump', dest='dump_csv', action='store_false', help='Skip csv dump')
parser.add_argument('--eval', nargs='+', default=[], help='Run Evaluations')

args = vars(parser.parse_args()); tryint = lambda s: int(s) if s.isdigit() else s
hm = [tryint(s) for s in args.pop('heatmap')]
groupby = args.pop('groupby'); mergeon = args.pop('mergeon');
if len(hm): options['Heatmap'] = (('Model', hm), process_heatmap, plot_heatmap); args['metrics'].append('Heatmap') 

enames = ['Training', 'Shifted Obs', 'Shifted Obs 2', 'Shifted Goal']
def add_eval(e): options[enames[e]] = (('Model', e), process_eval, plot_eval); args['metrics'].append(enames[e])
ev = [add_eval(tryint(e)) for e in args.pop('eval')]
metrics = [(metric, *options[metric]) for metric in args.pop('metrics')]
titles, scalars, procs, plotters = zip(*metrics)
label_exclude =  [] if args['alg'] and args['alg'] == 'DIRECT' else ['chi', 'omega', 'kappa']

experiments = fetch_experiments(**args, metrics=list(zip(titles, scalars)))
experiments = group_experiments(experiments, groupby, label_exclude, mergeon)
metrics = calculate_metrics(experiments, list(zip(titles, procs)))
figures = generate_figures(metrics, dict(zip(titles, plotters)))

# Save figures
out = f'{args["base"]}/plots/{"-".join(groupby)}'; os.makedirs(out, exist_ok=True)
if len(hm): os.makedirs(out+'/Heatmaps', exist_ok=True)
if len(ev): os.makedirs(out+'/Evaluation', exist_ok=True)
print("Done Evaluating. Saving Plots.")
for name, figure in tqdm(figures.items()): figure.write_image(f'{out}/{name}.pdf')