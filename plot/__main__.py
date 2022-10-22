"""
Example commands:
python -m plot plot/results/reload -b plot/results/train --eval
python -m plot plot/results/train 
"""
import argparse; import os; 
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
parser.add_argument('base', default='./results', help='The results root')
parser.add_argument('-a', dest='alg', help='Algorithm to vizualise.')
parser.add_argument('-b', dest='baseline', help='Base path of reloaded model.')
parser.add_argument('-e', dest='env', help='Environment to vizualise.')
parser.add_argument('-g', dest='groupby', nargs='+', default=[], metavar="groupby", help='Experiment keys to group plotted data by.')
parser.add_argument('-m', dest='metrics', nargs='+', default=[], choices=options.keys(), help='Experiment keys to group plotted data by.')
parser.add_argument('--heatmap', nargs='+', default=[], help='Environment to vizualise.')
parser.add_argument('--mergeon', help='Key to merge experiments e.g. algorithm.')
parser.add_argument('--no-dump', dest='dump_csv', action='store_false', help='Skip csv dump')
parser.add_argument('--eval', nargs='+', default=[], help='Run Evaluations')

tryint = lambda s: int(s) if s.isdigit() else s
args = vars(parser.parse_args()) 
hm = [tryint(s) for s in args.pop('heatmap')]
groupby = args.pop('groupby'); mergeon = args.pop('mergeon');
if len(hm): options['Heatmap'] = (('Model', hm), process_heatmap, plot_heatmap); args['metrics'].append('Heatmap') 

enames = ['Evaluation Training', 'Evaluation Shifted Obs', 'Evaluation Shifted Obs 2', 'Evaluation Shifted Goal']
def add_eval(e): options[enames[e]] = (('Model', e), process_eval, plot_eval); args['metrics'].append(enames[e])
[add_eval(tryint(e)) for e in args.pop('eval')]

metrics = [(metric, *options[metric]) for metric in args.pop('metrics')]
titles, scalars, procs, plotters = zip(*metrics)
label_exclude =  [] if args['alg'] and args['alg'] == 'DIRECT' else ['chi', 'omega', 'kappa']

# Load, sort and group experiments, calculate metrics and generate figures
experiments = fetch_experiments(**args, metrics=list(zip(titles, scalars)))
experiments = group_experiments(experiments, groupby, label_exclude, mergeon)
metrics = calculate_metrics(experiments, list(zip(titles, procs)))
figures = generate_figures(metrics, dict(zip(titles, plotters)))

# Save figures
out = f'{args["base"]}/plots/{"-".join(groupby)}'; os.makedirs(out, exist_ok=True)
if len(hm): os.makedirs(out+'/Heatmaps', exist_ok=True)
[figure.write_image(f'{out}/{name}.pdf') for name, figure in figures.items()]