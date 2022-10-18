from safety_env.plotting import heatmap_3D
import numpy as np; import plotly.graph_objects as go

# Helper functions to create scatters/graphs from experiment & metric
def plot_box(title, plot):
  box = lambda g: go.Box(name=g['label'], y=g['data'], marker_color=color(g['hue']), boxmean=True)  #boxmean='sd'
  return { 'layout': layout(title, False), 'data': [ box(g) for g in plot['graphs'] ] }


def smooth(data, degree=4):
  triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))
  smoothed = [np.sum(data[i:i + len(triangle)] * triangle)/np.sum(triangle) for i in range(degree, len(data) - degree * 2)]
  return [smoothed[0]] * int(degree + degree/2) + smoothed + [smoothed[-1] for _ in range(len(data)-len(smoothed) - int(degree + degree/2))]


def plot_ci(title, plot):
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=smooth(data), **kwargs)
  getmean = lambda g: scatter(g['data'][0], name=g['label'], mode='lines', line={'color': color(g['hue'])})
  getconf = lambda g: scatter(g['data'][1], fillcolor=color(g['hue'], True), fill='toself', line={'color': 'rgba(255,255,255,0)'}, showlegend=False)
  return { 'layout': layout(title), 'data': [getconf(g) for g in plot['graphs']] + [getmean(g) for g in plot['graphs']] }


def plot_heatmap(title, plot): (data, (vmin,vmax)) = plot['data']; return heatmap_3D(data, vmin, vmax)


#                      Red         Orange     Green          Blue        Purple 
# color_lookup = {"PPO": 340, "A2C": 40, "SIL": 130, "DIRECT": 210, "DQN": 290 }
def color(hue, sec=False): return 'hsva({},{}%,{}%,{})'.format(hue, 90-sec*20, 80+sec*20, 1.0-sec*0.8)


def layout(title=None, legend=True, wide=True): 
  axis = {'gridcolor': 'rgba(64, 64, 64, 0.32)', 'linecolor': 'rgb(64, 64, 64)',
    'mirror':True, 'ticks':'outside', 'showline':True, 'zeroline': True, 'zerolinecolor': 'rgba(64, 64, 64, 0.32)'} #'tickmode': 'linear', 'range':[-0.5,max(data.shape)-0.5], 
   
  return go.Layout( title=title, showlegend=legend, font=dict(size=24), 
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=900+300*wide+300*legend, height=600, xaxis=axis, yaxis=axis, plot_bgcolor='rgba(64,64,64,0.04)') #, paper_bgcolor='rgba(0,0,0,0)', 


def generate_figures(plots, generator):
  title = lambda plot: f'{plot["metric"]} ({plot["title"]})'
  return { title(p): go.Figure(**generator[p['metric']](title(p), p)) for p in plots}
