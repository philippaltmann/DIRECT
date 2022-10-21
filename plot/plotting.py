from safety_env.plotting import heatmap_3D
import numpy as np; import plotly.graph_objects as go

title = lambda plot: [plot["metric"], plot["title"]]

# Helper functions to create scatters/graphs from experiment & metric
def plot_box(plot):
  box = lambda g: go.Box(name=g['label'], y=g['data'][0], marker_color=color(g['hue']), boxmean=True) 
  figure = go.Figure(layout=layout(**dict(zip(['y','title'], title(plot)))), data=[box(g) for g in plot['graphs']])
  figure.add_hline(y=plot['graphs'][0]['data'][1], line_dash = 'dash', line_color = 'rgb(64, 64, 64)')
  return {' '.join(title(plot)): figure}


def smooth(data, degree=4):
  triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))
  smoothed = [np.sum(data[i:i + len(triangle)] * triangle)/np.sum(triangle) for i in range(degree, len(data) - degree * 2)]
  return [smoothed[0]] * int(degree + degree/2) + smoothed + [smoothed[-1] for _ in range(len(data)-len(smoothed) - int(degree + degree/2))]


def plot_ci(title, plot):
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=smooth(data), **kwargs)
  getmean = lambda g: scatter(g['data'][0], name=g['label'], mode='lines', line={'color': color(g['hue'])})
  getconf = lambda g: scatter(g['data'][1], fillcolor=color(g['hue'], True), fill='toself', line={'color': 'rgba(255,255,255,0)'}, showlegend=False)
  figure = go.Figure(layout=layout(title), data=[getconf(g) for g in plot['graphs']] + [getmean(g) for g in plot['graphs']])
  figure.add_hline(y=plot['graphs'][0]['data'][2], line_dash = 'dash', line_color = 'rgb(64, 64, 64)')
  return figure


def plot_heatmap(title, plot): (data, (vmin,vmax)) = plot['data']; return heatmap_3D(data, vmin, vmax)


#                      Red         Orange     Green          Blue        Purple 
# color_lookup = {"PPO": 340, "A2C": 40, "SIL": 130, "DIRECT": 210, "DQN": 290 }
def color(hue, sec=False): return 'hsva({},{}%,{}%,{})'.format(hue, 90-sec*20, 80+sec*20, 1.0-sec*0.8)


def layout(title=None, legend=True, wide=True, x='', y=''): 
  axis = lambda title: {'gridcolor': 'rgba(64, 64, 64, 0.32)', 'linecolor': 'rgb(64, 64, 64)', 'title': title,
    'mirror':True, 'ticks':'outside', 'showline':True, 'zeroline': True, 'zerolinecolor': 'rgba(64, 64, 64, 0.32)'} 
    #'tickmode': 'linear', 'range':[-0.5,max(data.shape)-0.5], 
   
  return go.Layout( title=title, showlegend=legend, font=dict(size=24), 
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=900+300*wide+300*legend, height=600, 
    xaxis=axis(x), yaxis=axis(y), plot_bgcolor='rgba(64,64,64,0.04)') #, paper_bgcolor='rgba(0,0,0,0)', 


def generate_figures(plots, generator): return { k:v for p in plots for k,v in generator[p['metric']](p).items()}
