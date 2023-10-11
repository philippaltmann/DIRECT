import argparse; import time; import os; 
from plot.metrics import *
from plot.plotting import * 

options = {k+p: v for k,v in { # Title, ScalarTag, process(scalar)->data, display(data)->trace
  'Training': ('rewards/return-100-mean', process_return, plot_return),
  'Validation': ('rewards/validation', process_return, plot_return),
  'Heatmap': ('Model', process_heatmap, plot_heatmap),
  'Buffer': ('buffer/momentum', process_buffer, plot_buffer),
  'Eval': ('Model', process_eval, plot_eval),
}.items() for p in ['','-pre']}

# Process commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument('base', help='The results root')
parser.add_argument('-a', dest='alg', nargs='+', help='The algorithm to vizualise') #choices=[*ALGS]
parser.add_argument('-e', dest='env', help='Environment to vizualise.')
parser.add_argument('-g', dest='groupby', nargs='+', default=[], metavar="groupby", help='Experiment keys to group plotted data by.')
parser.add_argument('-m', dest='metrics', nargs='+', default=['Buffer-pre', 'Heatmap-pre', 'Eval-pre', 'Training'], choices=options.keys(), help='Metrics to plot, optionally postfixed with -pre.')
parser.add_argument('--no-dump', dest='dump_csv', action='store_false', help='Skip csv dump')

args = vars(parser.parse_args()); tryint = lambda s: int(s) if s.isdigit() else s
if args['alg']: args['alg'] = ' '.join(args['alg'])
groupby = args.pop('groupby')

metrics = [(metric, *options[metric]) for metric in args.pop('metrics')]
titles, scalars, procs, plotters = zip(*metrics)

experiments = fetch_experiments(**args, metrics=list(zip(titles, scalars)))
experiments = group_experiments(experiments, ['env', *groupby])
experiments = calculate_metrics(experiments, list(zip(titles, procs)))
figures = generate_figures(experiments, dict(zip(titles, plotters)))

# Save figures
out = f'{args["base"]}/plots/{"-".join(groupby)}'; os.makedirs(out, exist_ok=True)
print("Done Evaluating. Saving Plots.")
for name, figure in tqdm(figures.items()): 
  os.makedirs(f"{out}/{name[:name.rindex('/')+1]}", exist_ok=True)  
  figure.write_image(f'{out}/{name}.{"png" if "Heatmap" in name else "svg"}') 