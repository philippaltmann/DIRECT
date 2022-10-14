import plotly.graph_objects as go

# Helper functions to create scatters/graphs from experiment & metric
def plot_box(title, plot):
  box = lambda g: go.Box(name=g['label'], y=g['data'], marker_color=color(g['hue']), boxmean=True)  #boxmean='sd'
  return { 'layout': layout(title, False), 'data': [ box(g) for g in plot['graphs'] ] }

# reward_threshold = {'type':'line', 'x0':0, 'y0':0.945, 'x1':1000000, 'y1':0.945, 'line':{'color':'#424242', 'width':2, 'dash':'dash'}}

def plot_ci(title, plot):
  # TODOs: smotthen, add reward threshold 
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=data, **kwargs)
  getmean = lambda g: scatter(g['data'][0], name=g['label'], mode='lines', line={'color': color(g['hue']), 'smoothing': 1.0})
  getconf = lambda g: scatter(g['data'][1], fillcolor=color(g['hue'], True), fill='toself', line={'color': 'rgba(255,255,255,0)'}, showlegend=False)
  return { 'layout': layout(title), 'data': [getconf(g) for g in plot['graphs']] + [getmean(g) for g in plot['graphs']] }


#                      Red         Orange     Green          Blue        Purple 
# color_lookup = {"PPO": 340, "A2C": 40, "SIL": 130, "DIRECT": 210, "DQN": 290 }
def color(hue, sec=False): return 'hsva({},{}%,{}%,{})'.format(hue, 90-sec*20, 80+sec*20, 1.0-sec*0.8)


def layout(title=None, legend=True, wide=True): 
  return go.Layout( title=title, showlegend=legend, font=dict(size=24), 
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=900+300*wide+300*legend, height=600)


def generate_figures(plots, generator):
  title = lambda plot: f'{plot["metric"]} ({plot["title"]})'
  return { title(p): go.Figure(**generator[p['metric']](title(p), p)) for p in plots}
