import argparse; import time; import os; 
from plot.metrics import *
from plot.plotting import * 

options = { # Title, ScalarTag, process(scalar)->data, display(data)->trace
  ###########
  # General #
  ###########

  'Training': ('rewards/return-100-mean', process_return, plot_return),


  ##############
  # Evaluation #
  ##############

  # Buffer Size  
  'Buffer': ('buffer/momentum', process_buffer, plot_buffer), # Histogram (final)
  'Buffers': ('buffer/momentum', process_buffer, plot_buffer), # Histograms (all)
  'Momentum': ('buffer/momentum', process_metric, plot_ci), # Buffer Momentum 
  'Scores': ('rewards/buffer/scores-mean', process_metric, plot_ci), # Buffer Scores  

  # Update Rate 
  'Discriminator': ('discriminator/acccuracy', process_metric, plot_ci),

  # Reward Mixture
  'DIRECT': ('rewards/direct-100-mean', process_direct, plot_direct), # DIRECT Reward (heatmap & progress)
  

  #############
  # Benchmark #
  #############

  'Shift': ('rewards/evaluation-0', process_shift, plot_shift), # Evaluation Policy Action heatmap & progress

  # Distributional Shift:
  'Evaluation': ('rewards/evaluation-0', process_return, plot_return),

  # Previous unused pots 
  # 'Eval': ('Model', process_eval, plot_eval),
}

# Process commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument('base', help='The results root')
parser.add_argument('-a', dest='alg', nargs='+', help='The algorithm to vizualise') #choices=[*ALGS]
parser.add_argument('-e', dest='env', help='Environment to vizualise.')
parser.add_argument('-g', dest='groupby', nargs='+', default=[], metavar="groupby", help='Experiment keys to group plotted data by.')
parser.add_argument('-m', dest='metrics', nargs='+', default=['Training'], choices=options.keys(), help='Metrics to plot.')
parser.add_argument('--merge', nargs='+', default=[], help='Additional Metric to merge into plot')
parser.add_argument('--no-dump', dest='dump_csv', action='store_false', help='Skip csv dump')

args = vars(parser.parse_args()); tryint = lambda s: int(s) if s.isdigit() else s
if args['alg']: args['alg'] = ' '.join(args['alg'])
groupby = args.pop('groupby')
if len(merge := args.pop('merge')): args['metrics'].extend(merge)

metrics = [(metric, *options[metric]) for metric in args.pop('metrics')]
titles, scalars, procs, plotters = zip(*metrics)

experiments = fetch_experiments(**args, metrics=list(zip(titles, scalars)))
experiments = group_experiments(experiments, ['env', *groupby])
experiments = calculate_metrics(experiments, list(zip(titles, procs)))
figures = generate_figures(experiments, dict(zip(titles, plotters)), merge=merge)

# Save figures
out = f'{args["base"]}/plots/{"-".join(groupby)}'; os.makedirs(out, exist_ok=True)
print("Done Evaluating. Saving Plots.")
for name, figure in tqdm(figures.items()): 
  os.makedirs(f"{out}/{name[:name.rindex('/')+1]}", exist_ok=True)  
  go.Figure().write_image(f'{out}/{name}.pdf', format="pdf")
  time.sleep(1) # Await package loading to aviod warning boxes
  figure.write_image(f'{out}/{name}.pdf', format="pdf")