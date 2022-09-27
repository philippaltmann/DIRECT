"""
Example commands:
python -m vizualisation experiments/hyperparameters/sa -m kappa
"""
import argparse; import os; from parse import parse
import numpy as np; import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common.plotting import fetch_experiments

# Process commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument('base', default='./results', help='The results root')
parser.add_argument('-a', dest='alg', help='Algorithm to vizualise.')
parser.add_argument('-e', dest='env', help='Environment to vizualise.')
parser.add_argument('-m', dest='mode', default='benchmark', help='Mode. (Benchmark for algorithm comparison or name of hp)')


args = parser.parse_args()

print(args.mode)

# Metrics to be plotted (=> one plot per entry \w one graph per experiment)
metrics = [
  ('Reward', 'metrics/validation_reward'),
  # ('Return', 'raw/rewards_return-100')
]
#                      Red         Orange     Green          Blue        Purple 
color_lookup = {"PPO": 340, "A2C": 40, "SIL": 130, "DIRECT": 210, "DQN": 290 }

# Fetch experiment logfiles
experiments = fetch_experiments(args.base, args.alg, args.env)


# Helper function to calculate mean and confidence interval for list of DataFrames
def data2ci(data):
  training_steps = [d.index[-1] for d in data]
  for d in data: d.loc[np.max(training_steps)] = float(d.tail(1)['Data'])
  mean = pd.concat(data, axis=1, sort=False).bfill().mean(axis=1)
  std = pd.concat(data, axis=1, sort=False).bfill().std(axis=1)
  confidence = pd.concat([mean+0.5*std, (mean-0.5*std).iloc[::-1]])
  return {'mean': mean, 'confidence': confidence}


# Helper to convert tb Scalar data to pd DataFrame & pack Dataframes for given metrics 
columns, index, exclude = ['Walltime', 'Step', 'Data'], 'Step', ['Walltime']
extract_data = lambda data: pd.DataFrame.from_records(data, columns=columns, index=index, exclude=exclude)
calc_metrics = lambda experiment, key: data2ci([extract_data(tb.Scalars(key)) for tb in experiment['tb']])
process_data = lambda experiment: { name: calc_metrics(experiment, key) for name, key in metrics }

# Process given experiments
experiments = [{**exp, 'data': process_data(exp) } for exp in experiments]

# Generlate list of Envornments -> 1 plot per env&metric \w 1line per alg avg over all runs
envs = list(dict.fromkeys([exp['env'] for exp in experiments]))

print('Done aggregating data')


# print(experiments)
hp_format = {'chi': 'Χ: {:.1f}', 'omega':  'ω: {:.1f}', 'kappa': 'κ: {:d}'}
hp_lookup = lambda e: ' ('+', '.join([f.format(e[key]) for key,f in hp_format.items() if key != args.mode])+')'
label = lambda e: e["algorithm"] + hp_lookup(e) if "DIRECT" in e["algorithm"] else ""

# Generate Plots 
if args.mode == 'benchmark':
  color = lambda e, sec=False: 'hsva({},{}%,{}%,{}%)'.format(color_lookup[e["algorithm"]], 90-sec*20, 80+sec*20, 100-sec*80)
  plots = [{'title': f'{env} {metric}', 
      'mean': [ go.Scatter( x=exp['data'][metric]['mean'].index, y=exp['data'][metric]['mean'], 
          name=label(exp), line={'color':color(exp)}, mode='lines') 
        for exp in experiments if env == exp['env']], 
      'confidence': [ go.Scatter( x=exp['data'][metric]['confidence'].index, y=exp['data'][metric]['confidence'], 
          fillcolor=color(exp, True), fill='toself', line=dict(color='rgba(255,255,255,0)'), showlegend=False) 
        for exp in experiments if env == exp['env']],
    } for metric, _ in metrics for env in envs 
  ]
else: 
  print('else')
  #args.mode contains param to iter 
  # get all options 
  options = list(dict.fromkeys([exp[args.mode] for exp in experiments]))

  plots = [{'title': f'{env} {name} {hp_format[args.mode].format(option)}', 
      'mean': [ go.Scatter( x=exp['data'][name]['mean'].index, y=exp['data'][name]['mean'], 
          name=label(exp), mode='lines') 
        for exp in experiments if env == exp['env'] and option == exp[args.mode]], 
      'confidence': [ go.Scatter( x=exp['data'][name]['confidence'].index, y=exp['data'][name]['confidence'], 
          fill='toself', line=dict(color='rgba(255,255,255,0)'), showlegend=False) 
        for exp in experiments if env == exp['env'] and option == exp[args.mode]]
    } for name, _ in metrics for env in envs for option in options
  ]

print('Done creating plots')


# print([tb.Tags().keys() for e in experiments for tb in e['tb']])
# print([tb.Tags()['scalars'] for e in experiments for tb in e['tb']])
# print([np.array(tb.Scalars('raw/rewards_return-100')).shape for e in experiments for tb in e['tb']])
# print([tb.Scalars('raw/rewards_return-100') for e in experiments for tb in e['tb']])
# print([tb.Scalars('raw/rewards_return-100')['step'] for e in experiments for tb in e['tb']])
# print([pd.DataFrame.from_records(tb.Scalars('raw/rewards_return-100'), index='step', exclude='wall_time') for e in experiments for tb in e['tb']])
dest = f'{args.base}/plots'
os.makedirs(dest, exist_ok=True)

reward_threshold = {'type':'line', 'x0':0, 'y0':0.945, 'x1':1000000, 'y1':0.945, 'line':{'color':'#424242', 'width':2, 'dash':'dash'}}

# Training Reward
layout = go.Layout(margin=dict(l=8, r=8, t=8, b=8), width=1500, height=600, font=dict(size=24)) #showlegend=False

# Training Episodes
# go.Layout( margin=dict(l=8, r=8, t=8, b=8), width=1000, height=400, font=dict(size=24), xaxis=dict(tickfont=dict(size=32))


# , shapes=[reward_threshold]
plots = [go.Figure(data=plot['confidence'] + plot['mean'], layout=layout).write_image(f'{dest}/{plot["title"]}.pdf') for plot in plots]
# training_reward.write_image("figures/bench-training_rewards.pdf")
# print(rewards)
# print(training_steps)

# print([e['tb'].Tags().keys() for e in experiments])
# print([e['tb'].Tags()['scalars'] for e in experiments])
# print([e['tb'].Scalars('raw/rewards_return-100') for e in experiments])
# print([np.array(e['tb'].Scalars('raw/rewards_return-100')).shape for e in experiments])
# event_acc
# print(experiments)