from safety_env.plotting import heatmap_3D
import numpy as np; import plotly.graph_objects as go
from plotly.subplots import make_subplots


title = lambda plot, y=None: [y or plot["metric"], plot["title"]]

# Helper functions to create scatters/graphs from experiment & metric
def plot_box(plot, y=None):
  box = lambda g: go.Box(name=g['label'], y=g['data'][0], marker_color=color(g['hue']), boxmean=True) 
  figure = go.Figure(layout=layout( y=y or plot['metric']), data=[box(g) for g in plot['graphs']])
  # **dict(zip(['y','title'], title(plot,y)))
  figure.add_hline(y=plot['graphs'][0]['data'][1], line_dash = 'dash', line_color = 'rgb(64, 64, 64)')
  return {' '.join(title(plot)): figure}


def plot_pie(data):
  labels = [str(l).split('.')[-1] for l in data[0]['data'].keys()]; titles = [ d['label'] for d in data]
  colors = {'colors':[color({'GOAL': 140, 'TIME': 60, 'FAIL': 350}[l]) for l in labels]}
  figure = make_subplots(rows=1, cols=len(data), subplot_titles=titles, specs=[[{'type':'domain'}]*len(data)])
  [figure.add_trace(go.Pie(labels=labels, values=list(d['data'].values()), name=d['label'], hole=.4, marker=colors), 1, i+1) for i,d in enumerate(data)]
  figure.update_layout(margin=dict(l=8, r=8, t=8, b=8), width=600, height=200) #annotations=titles, 
  return figure   


def smooth(data, degree=4):
  triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))
  smoothed = [np.sum(data[i:i + len(triangle)] * triangle)/np.sum(triangle) for i in range(degree, len(data) - degree * 2)]
  return [smoothed[0]] * int(degree + degree/2) + smoothed + [smoothed[-1] for _ in range(len(data)-len(smoothed) - int(degree + degree/2))]


def plot_ci(plot):
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=smooth(data), **kwargs)
  getmean = lambda g: scatter(g['data'][0], name=g['label'], mode='lines', line={'color': color(g['hue'])})
  getconf = lambda g: scatter(g['data'][1], fillcolor=color(g['hue'], True), fill='toself', line={'color': 'rgba(255,255,255,0)'}, showlegend=False)
  data = [getconf(g) for g in plot['graphs']] + [getmean(g) for g in plot['graphs']]
  figure = go.Figure(layout=layout( y='Mean Return', x='Timesteps', legend=True), data=data)
  # figure = go.Figure(layout=layout(**dict(zip(['y','title'], title(plot))), x='Timesteps', legend=False), data=data)
  figure.add_hline(y=plot['graphs'][0]['data'][2], line_dash = 'dash', line_color = 'rgb(64, 64, 64)')
  return {' '.join(title(plot)): figure}


def plot_eval(plot):
  box = list(plot_box({**plot, 'graphs': [{**g, 'data': g['data'][:2]}for g in plot['graphs']]}, 'Mean Return').values())[0]
  pie = plot_pie([{'label': g['label'], 'data': g['data'][2]}for g in plot['graphs']])
  _t = lambda type: f"Evaluation/{type} {plot['metric']} ({plot['title']})"
  return { _t('Termination'): pie,_t('Reward'): box }
  # plot['title'] =''#= plot['title'] + plot['metric']; plot['metric'] = 'Reward'; 
  return dict(zip([plot['metric']],plot_box(plot, 'Mean Return').values()))


def plot_heatmap(plot): return {f'Heatmaps/{plot["title"]}': heatmap_3D(plot['data'], compress=True)}


#                      Red         Orange     Green          Blue        Purple 
# color_lookup = {"PPO": 340, "A2C": 40, "SIL": 130, "DIRECT": 210, "DQN": 290 }
def color(hue, sec=False): return 'hsva({},{}%,{}%,{})'.format(hue, 90-sec*20, 80+sec*20, 1.0-sec*0.8)


def layout(title=None, legend=True, wide=True, x='', y=''): 
  axis = lambda title: {'gridcolor': 'rgba(64, 64, 64, 0.32)', 'linecolor': 'rgb(64, 64, 64)', 'title': title,
    'mirror':True, 'ticks':'outside', 'showline':True, 'zeroline': True, 'zerolinecolor': 'rgba(64, 64, 64, 0.32)'} 
    #'tickmode': 'linear', 'range':[-0.5,max(data.shape)-0.5], 
  return go.Layout( title=title, showlegend=legend, font=dict(size=24), 
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=600+200*wide+100*legend, height=400, 
    xaxis=axis(x), yaxis=axis(y), plot_bgcolor='rgba(64,64,64,0.04)') #, paper_bgcolor='rgba(0,0,0,0)', 
   
  return go.Layout( title=title, showlegend=legend, font=dict(size=24), 
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=900+300*wide+300*legend, height=600, 
    xaxis=axis(x), yaxis=axis(y), plot_bgcolor='rgba(64,64,64,0.04)') #, paper_bgcolor='rgba(0,0,0,0)', 


def generate_figures(plots, generator): return { k:v for p in plots for k,v in generator[p['metric']](p).items()}
