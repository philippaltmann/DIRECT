import math; import plotly.graph_objects as go

LAYOUT = {
  'Buffer-pre': {'x': 'Pre-training Steps', 'y': 'Buffer Momentum', 'width': 500, 'legend': False},
  'Eval-pre': {'y': 'Validation Return', 'width': 600, 'legend': False},
  'Eval': {'y': 'Validation Return', 'width': 600, 'legend': False},
  'Training': {'x': 'Steps', 'y': 'Mean Return', 'width': 800, 'legend': True}
}

RT = {
  'Maze9Target': [33.875]*2,
  'Maze13Target': [21.575]*2,
  'Maze15Target': [52.3]*2,

  'PointMaze11Target': [131.625]*2,
  'PointMaze13Target': [165.75]*2,
  'PointMaze15Target': [123.0]*2
}

AXES = {
  'Maze9Target': {'x': {'range': [0,150000], 'dtick': 50000}, 'y': [-100,50]},
  'Maze13Target': {'x': {'range': [0,150000], 'dtick': 50000}, 'y': [-100,50]},
  'Maze15Target': {'x': {'range': [0,150000], 'dtick': 50000}, 'y': [-200,100]},

  'PointMaze11Target': {'x': {'range': [0,500000], 'dtick': 100000}, 'y': [-600.0, 300.0]},
  'PointMaze13Target': {'x': {'range': [0,500000], 'dtick': 100000}, 'y': [-900,450]},
  'PointMaze15Target': {'x': {'range': [0,500000], 'dtick': 100000}, 'y': [-1200.0, 600.0]}
}

title = lambda plot, y=None: [plot["title"], plot["metric"]]
traceorder = lambda key: [i for k,i in {'GRASP': 0, 'GAIN': 1, 'PPO': 2, 'SAC': 3}.items() if k in key][0]

# Helper functions to create scatters/graphs from experiment & metric
def plot_ci(plot, xmax=None):  
  env = plot['title']# plot['env'].unwrapped.name 
  plot['graphs'].sort(key=lambda g: traceorder(g['label'])) # Sort traces  
  smooth = {'shape':  'spline',  'smoothing': 0.4}
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=data, **kwargs)
  dash = lambda g: {'dash': 'dash'} if 'Evaluation' in plot['metric'] else {}
  getmean = lambda g: scatter(g['data'][0], name=g['label'], mode='lines', line={'color': color(g), **smooth,  **dash(g)})
  getconf = lambda g: scatter(g['data'][1], fillcolor=color(g, 1), fill='toself', line={'color': 'rgba(255,255,255,0)', **smooth}, showlegend=False)
  data = [getconf(g) for g in plot['graphs']] + [getmean(g) for g in plot['graphs']] #+ threshold
  if plot['env'].info['reward_threshold'] is not None and plot['metric'] == 'Training':
    data += [go.Scatter(y=RT[env], 
                        x=[0,max([g['data'][0].tail(1).index[0] for g in plot['graphs']])],
                        name='Solved', mode='lines', line={'dash':'dot', 'color':'rgb(64, 64, 64)'})] 
  figure = go.Figure(layout=layout(**LAYOUT[plot["metric"]], inset=len(data)<18), data=data)
  # xmax = xmax or int(math.floor(max([g['data'][0].index[-1] for g in plot['graphs']])/10))*10
  if plot['metric'] == 'Training':
    figure.add_vrect(x0=0, x1=42*2048*4/8, line_width=0, fillcolor="black", opacity=0.05)

    figure.update_xaxes(**AXES[env]['x'], tickmode='linear') 
    figure.update_yaxes(range=AXES[env]['y']) #plot['env'].reward_range, tickmode = 'linear'

  return {'/'.join(title(plot)): figure}

def plot_return(data): return plot_ci(data) # , xmax={7: 150000, 15: 150000, 9: 500000, 11: 500000}[data['env'].size[0]]

def plot_buffer(data): return {**plot_ci(data['momentum']), **plot_heatmap(data['heatmap'], 'Heatmap')}


def plot_eval(plot, y=None):
  box = lambda g: go.Box(name=g['label'].replace('-','<br>'), y=g['data'], marker_color=color(g), boxmean=True) 
  plot['graphs'].sort(key=lambda g: traceorder(g['label'])) # Sort traces  
  figure = go.Figure(layout=layout(**LAYOUT[plot["metric"]]), data=[box(g) for g in plot['graphs'][::-1]])
  # figure.update_yaxes(range=plot['env'].reward_range) #, tickmode = 'linear'
  # figure.add_hline(y=plot['graphs'][0]['data'][1], line_dash = 'dash', line_color = 'rgb(64, 64, 64)')
  return {'/'.join(title(plot)): figure}

def plot_heatmap(plot, label=None): 
  _s = lambda d: int(math.sqrt(len(d)))
  reshape = lambda d: list(d.to_numpy().reshape((_s(d),)*2)[::-1,:])
  axis = dict(showticklabels = False)
  def heatmap (g): 
    fig = go.Figure(
      layout=dict(width=400, height=400, xaxis=axis, yaxis=axis, margin=dict(l=0, r=0, t=0, b=0)), 
      data=go.Heatmap(z=reshape(g['data']),colorscale=[(0,'#000000'),(1,'#0080C0')]))  #[(0,'#0080C0'),(1,'#00C060')]
    # fig.update(layout_showlegend=False)
    # fig.update_coloraxes(showscale=False)
    fig.update_traces(showscale=False)
    return fig  
  return {'/'.join(title(plot))+f'-{label or g["label"]}': heatmap(g) for g in plot['graphs']}
  return {f"hm/{''.join(title(plot))}{g['label']}": heatmap(g) for g in plot['graphs']}

def color(g, dim=0): 
  hue = lambda alg: [hue for key,hue in {'GAIN': 150, 'GRASP': 200, 'TODO': 230, 'SAC':40, 'PPO': 350, 'TODO': 70}.items() if key in alg][0]
  return 'hsva({},{}%,{}%,{:.2f})'.format(hue(g['label']), 90-dim*20, 80+dim*20, 1.0-dim*0.8)

def layout(title=None, legend=True, wide=True, x='', y='', inset=False, width=None):
  width = width or 600+200*wide+100*legend 
  d,m,l = 'rgb(64, 64, 64)', 'rgba(64, 64, 64, 0.32)', 'rgba(64,64,64,0.04)'
  axis = lambda title: {'gridcolor': m, 'linecolor': d, 'title': title, 'mirror':True, 'ticks':'outside', 'showline':True, 'zeroline': True, 'zerolinecolor': m} 

  return go.Layout( title=title, showlegend=legend, font=dict(size=20),  
    legend={'yanchor':'top', 'y':0.935, 'xanchor':'left', 'x':0.01,'bgcolor':l,'bordercolor':d,'borderwidth':1} if inset else {},
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=width, height=400, 
    xaxis=axis(x), yaxis=axis(y), plot_bgcolor=l) #, paper_bgcolor='rgba(0,0,0,0)', 
   
def generate_figures(plots, generator): return { k:v for p in plots for k,v in generator[p['metric']](p).items()}
