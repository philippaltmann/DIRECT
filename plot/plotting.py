import numpy as np; import plotly.graph_objects as go
from scipy.stats import percentileofscore
from plotly.subplots import make_subplots
from .metrics import HPS


LAYOUT = {
  'Training': {'x': 'Epoch', 'y': 'Mean Return', 'width': 800, 'legend': True},
  'Evaluation': {'x': 'Epoch', 'y': 'Evaluation Return', 'width': 800, 'legend': True},
  'Discriminator': {'x': 'Epoch', 'y': 'Discriminator Accuracy', 'width': 800, 'legend': True},
  'DIRECT': {'x': 'Steps', 'y': 'DIRECT Return', 'width': 800, 'legend': True},
  'Momentum': {'x': 'Epoch', 'y': 'Buffer Momentum', 'width': 800, 'legend': True},
  'Scores': {'x': 'Epoch', 'y': 'Buffer Return', 'width': 800, 'legend': True},
  # 'Eval': {'y': 'Validation Return', 'width': 600, 'legend': False},
}


AXES = {
  # Env-specific axes
  'Maze9Sparse': {'x': {'range': [0,24], 'dtick': 4}, 'y': [-100,50]},
  'PointMaze9': {'x': {'range': [0,96], 'dtick': 4}, 'y': [-400,200]},
  'HolesTrain': {'x': {'range': [0,8], 'dtick': 4}, 'y': [-105,50]},
  'HolesShift': {'x': {'range': [0,8], 'dtick': 4}, 'y': [-105,50]},
  'FetchReach': {'x': {'range': [0,96], 'dtick': 4}, 'y': [-100,50]},

  # Metric-specific sub-axes
  'Momentum': {'y': {'range': [0, None], 'title': 'Buffer Momentum'}},
  'Discriminator': {'y': {'range': [0.,1.], 'dtick': 1., 'title': 'Discriminator Accuracy'}},
  'DIRECT': {'y': {'range': [0, 100], 'title': 'DIRECT Return'}},
}


title = lambda plot, y=None: [plot["title"], plot["metric"]]


def hp_choices(g):
  f = 4 if 'κ' in g and 'Point' in g['env'] else 1
  if len(hp := [p for p in HPS.keys() if p in g]): 
    return hp[0], [p*f for p in HPS[hp[0]]]
  return None


def traceorder(g):
  if (hp:=hp_choices(g)) is not None: return [i for i,k in enumerate(hp[1]) if k==g[hp[0]]][-1]
  return [i for i,o in enumerate(['DIRECT', 'GASIL', 'SIL', 'PPO', 'A2C', 'DQN']) if o in g['label']][-1]


def color(g, dim=0): 
  """
  40: Orange          GASIL
  70: Yellow          A2C
  150: Green          SIL
  200: Light Blue     DIRECT
  230: Blue           (DQN)
  350: Red            PPO
  """
  hue = lambda alg: [hue for key,hue in {'DIRECT': 200, 'GASIL': 40, 'SIL': 150, 'PPO': 350, 'A2C': 70, '':200}.items() if key in alg][0]
  if g['algorithm'] in g['label']: val = lambda alg: 80
  else: 
    hp, hps = hp_choices(g)
    val = lambda g: [val for val,p in zip([100,80,60,40,20],hps) if p==g[hp]][-1]
  return 'hsva({},{}%,{}%,{:.2f})'.format(hue(g['algorithm']), 90-dim*20, val(g)+dim*20, 1.0-dim*0.8)


# Helper functions to create scatters/graphs from experiment & metric
def plot_ci(plot, base=None, secondary=True):  
  if base is None and 'Training' in plot: base = plot_return(plot['Training'])
  env = plot['env'].unwrapped.name; plot['graphs'].sort(key=lambda g: traceorder(g)) # Sort traces  
  smooth = {'shape':  'spline',  'smoothing': 0.4}
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=data, **kwargs)
  dash = ['Evaluation', 'Buffer', 'Discriminator', 'DIRECT', 'Shift', 'Momentum']
  dash = {'dash': 'dash'} if any(m in plot['metric'] for m in dash) else {}
  if plot['metric'] == 'Scores': dash = {'dash':'dot'}
  getmean = lambda g: scatter(g['data'][0], name=g['label'], mode='lines', line={'color': color(g), **smooth,  **dash}, showlegend = base is None)
  getconf = lambda g: scatter(g['data'][1], fillcolor=color(g, 1), fill='toself', line={'color': 'rgba(255,255,255,0)', **smooth}, showlegend=False)
  data = []
  if base is None: data += [getconf(g) for g in plot['graphs']]
  data += [getmean(g) for g in plot['graphs']]

  spec = {"secondary_y": secondary} if secondary and base is not None else {}

  if base is not None: 
    figure = list(base.values())[0]; 
    if secondary: figure.set_subplots(specs=[[spec.copy()]])
    [figure.add_trace(d, **spec) for d in data]
  else: figure = go.Figure(layout=layout(**LAYOUT[plot["metric"]], inset=len(data)<18), data=data)
  
  figure.update_xaxes(**AXES[env]['x']); figure.update_yaxes(range=AXES[env]['y'])
  if plot["metric"] == 'DIRECT' and 'Point' in plot["env"].get_wrapper_attr('name'): spec = {**spec, 'range': [0, 400]}
  if plot["metric"] in AXES: figure.update_yaxes(**{**AXES[plot["metric"]]['y'], **spec})

  return {'/'.join(title(plot)): figure}


def plot_return(data): # For metric 'Training' only
  env =  data['env'].unwrapped.name
  plot = plot_ci(data); figure = list(plot.values())[0]
  return plot


def plot_histogram(plot, label=None): 
  _s = lambda d: int(np.sqrt(len(d))) #math.sqrt
  reshape = lambda d: list(d.to_numpy().reshape((_s(d),)*2)[::-1,:])
  d,m,l = 'rgb(64, 64, 64)', 'rgba(64, 64, 64, 0.32)', 'rgba(64,64,64,0.04)'
  axis = lambda max: dict(showticklabels=False, range=[0,max], title='', backgroundcolor=m, zerolinecolor=d, showgrid=False)
  scene = dict(xaxis=axis(9), yaxis=axis(9), zaxis=axis(None), aspectmode='cube', camera_center_z=-0.1)
  
  def histograms(g):
    n = len(g['data']); scenes = {f'scene{i+1}': scene for i in range(n)}    
    histogram = lambda data: go.Mesh3d(data, color=color(g), flatshading=True)
    fig = make_subplots(rows=1, cols=n, specs=[[{'type':'Mesh3D'} for _ in range(n)]])
    [fig.add_trace(histogram(data), col=1+i, row=1) for i, data in enumerate(g['data'])]
    fig.update_layout(width=600*n, height=600, margin=dict(l=0,r=0,t=0,b=0), **scenes)
    return fig
  return {'/'.join(title(plot))+f'-{label or g["label"]}': histograms(g) for g in plot['graphs']}


def plot_heatmap(plot, label=None, holes=False): 
  
  def heatmap(data, show_agent=False, compress=False):
    data = np.array(data)
    if compress: data = data[1:-1,1:-1] # Crop walls[[field for field in row[1:-1]] for row in data[1:-1]]
    rows,cols = data.shape[0:2]; hasnan = lambda a: any([np.isnan(i) for i in a])
    lookup = {'g':'hsl(140,100%,50%)', 'w': 'rgba(192,192,192,0.5)', 'h': 'rgba(0,0,0,0)'}
    # _heat = lambda v: 'hsl({:0.0f},80%,{:0.0f}%)'.format(pct(v)*140, 75-30*abs((pct(v)-1/3)*3/2)) # global heat 
    _heat = lambda v: 'hsl({:0.0f},80%,{:0.0f}%)'.format(v/100*140, 75-30*abs((v/100-1/3)*3/2)) # local heat (state only)
    color = lambda v: lookup[v] if type(v) == str and v in ['a','g','h','w'] else _heat(v) 
    _w = lambda c,r: not compress and (r in [0,rows-1] or c in [0,cols-1])
    _g = lambda c,r: r in [1-compress,rows-2+compress] and c in [1-compress,cols-2+compress] 
    t = lambda c,r: 'g' if _g(c,r) else 'w' if _w(c,r) or not holes else 'h'
    
    # Reference Point constructors for triangulations
    ct = lambda c,r,h=0: (c,r,h) # Helper points at c(ollumn), r(ow), h(ight) for w(idth)
    ul = lambda c,r,h=0,w=0.5: (c-w, r-w, h); ur = lambda c,r,h=0,w=0.5: (c+w, r-w, h)
    dr = lambda c,r,h=0,w=0.5: (c+w, r+w, h); dl = lambda c,r,h=0,w=0.5: (c-w, r+w, h)
    points = lambda c,r,v: [ct(c,r,0),ul(c,r,0),ur(c,r,0),dr(c,r,0),dl(c,r,0),ul(c,r,1),ur(c,r,1),dr(c,r,1),dl(c,r,1)]

    #  Reference Point constructors for positioning text
    uc = lambda c,r,w=0.3: (c, r-w, 0.1); rc = lambda c,r,w=0.3: (c+w, r, 0.1)
    dc = lambda c,r,w=0.3: (c, r+w, 0.1); lc = lambda c,r,w=0.3: (c-w, r, 0.1)
    labels = lambda c,r: [uc(c,r),lc(c,r),dc(c,r),rc(c,r)]
    txt = lambda c,r,val: [(*p, (f'{v:.0f}' if not np.isnan(v) else ''), 'black') for p,v in zip(labels(c,r), val)] # Percentage 

    # Indices of points to build triangle (up, right, down, left)
    _adapt = lambda c,r,t: tuple(map(lambda v: v+9*(r*cols+c),t))
    triang = lambda c,r,i: _adapt(c,r, [(0,1,2), (0,4,1), (0,3,4), (0,2,3)][i]) #urdl(0,1,2), (0,2,3), (0,3,4), (0,4,1)
    border = lambda c,r: [ul(c,r), dl(c,r), dr(c,r), ul(c,r), dl(c,r), ur(c,r), dr(c,r), ul(c,r), ur(c,r), dr(c,r)]
    square = lambda c,r: [_adapt(c,r,v) for v in [(1,2,3), (1,3,4), (6,7,8), (5,6,8), (1,2,6), (2,5,6), (2,3,6), (3,6,7), (3,4,8), (3,7,8), (1,4,8), (1,5,8)]]
    field = lambda c,r,val: [] if hasnan(val) else [(*triang(c,r,i), color(v)) for i,v in enumerate(val)]
    cube = lambda c,r,val: [(*f, color(t(c,r))) for f in square(c,r)] if hasnan(val) else [] 
    agent = lambda c,r,val: [(*f, 'hsl(210,100%,50%)') for f in square(c,r)] if show_agent and c==cols-2 and r==1 else []
    process = lambda *f: np.array([result for fn in f for r, row in enumerate(data) for c,val in enumerate(reversed(row)) for result in fn(c,r,val)]).T.tolist()

    p = dict(zip(['x','y','z'], process(points))); f = dict(zip(['i','j','k','facecolor'], process(field, cube, agent)))
    b = dict(zip(['x','y','z'], process(lambda c,r,val: [] if hasnan(val) else border(c,r))))
    l = dict(zip(['x','y','z','text','textfont'], process(txt))); l['textfont'] = {'color': l['textfont'], 'size':20} 

    mesh=go.Mesh3d(**p, **f) 
    lines = go.Scatter3d(**b, mode='lines', marker={'color':'black'})
    text=go.Scatter3d(**l, mode='text', textposition='middle center') 
    axis = {'showbackground':False, 'tickmode': 'linear', 'range':[-0.5,max(cols,rows)-0.5], 'visible': False} 
    scene = {'xaxis':axis, 'yaxis':axis, 'zaxis':axis, 'camera': {'eye': {'x':0, 'y':0, 'z':(data.shape[0]-1.2)/10}, 'up':{'x':0, 'y':-1, 'z':0}}} 
    if holes: 
      axis = lambda r: {'showbackground':False, 'tickmode': 'linear', 'range':r, 'visible': False} 
      scene = {**scene, 'xaxis':axis((-0.5,8.5)), 'yaxis':axis((-1.5,7.5)), 'zaxis':axis((0,7))}
    layout = go.Layout(margin={'l':0,'r':0,'t':0,'b':0}, width=cols*100, height=rows*100, scene=scene,  showlegend=False, paper_bgcolor='black') #title=title
    return go.Figure(data=[mesh,lines,text], layout=layout)
  return {'/'.join(title(plot))+f'-{label or g["label"]}': heatmap(g['data']) for g in plot['graphs']}


###################
# Aggregate Plots # 
###################

def plot_buffer(data): 
  base = None
  if 'Training' in data: base = plot_return(data['Training'])
  if 'Scores' in data: base = plot_ci(data['Scores'], base=base, secondary=False)
  if 'Momentum' in data: base = plot_ci(data['Momentum'], base=base, secondary=True)
  base = {} if base == None else {'/'.join(title(data['heatmap'])): list(base.values())[0]}
  return {**base, **plot_histogram(data['heatmap'])} 

def plot_direct(data):
  plot = data['direct']; 
  if 'Training' in data: plot['Training'] = data['Training']
  return {**plot_ci(plot), **plot_heatmap(data['heatmap'])}

def plot_shift(data):
  plot = data['shift']; 
  if 'Training' in data: plot['Training'] = data['Training']
  return {**plot_ci(plot, secondary=False), **plot_heatmap(data['heatmap'], holes=True)}


##############
# Generators #
##############

def layout(title=None, legend=True, wide=True, x='', y='', inset=False, width=None):
  width = width or 600+200*wide+100*legend 
  d,m,l = 'rgb(64, 64, 64)', 'rgba(64, 64, 64, 0.32)', 'rgba(64,64,64,0.04)'
  axis = lambda title: {'gridcolor': m, 'linecolor': d, 'title': title, 'mirror':True, 'ticks':'outside', 'showline':True, 'zeroline': True, 'zerolinecolor': m, 'titlefont': dict(size=24), 'tickfont': dict(size=16)} #'tickmode':'linear' 
  _legend = {'yanchor':'top', 'y':0.88, 'xanchor':'left', 'x':0.01,'bgcolor':l,'bordercolor':d,'borderwidth':1} 

  return go.Layout( title=title, showlegend=legend, font=dict(size=20), legend=(_legend if inset else {}),
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=width, height=400, 
    xaxis=axis(x), yaxis=axis(y), yaxis2=axis(''), plot_bgcolor=l) #, paper_bgcolor='rgba(0,0,0,0)', 
   

def generate_figures(plots, generator, merge=[]): 
  for merge in merge: 
    M = [plots.pop(i) for i in range(len(plots))[::-1] if plots[i]['metric']==merge]
    [p.update({merge: m}) for p in plots for m in M if p['env'] == m['env']]  
  # if merge is not None: M = plots.pop(-1); [p.update({merge: M}) for p in plots]
  return { k:v for p in plots for k,v in generator[p['metric']](p).items()}


##########
# Unused #
##########

def plot_eval(plot, y=None):
  box = lambda g: go.Box(name=g['label'].replace('-','<br>'), y=g['data'], marker_color=color(g), boxmean=True) 
  plot['graphs'].sort(key=lambda g: traceorder(g['label'])) # Sort traces  
  figure = go.Figure(layout=layout(**LAYOUT[plot["metric"]]), data=[box(g) for g in plot['graphs'][::-1]])
  return {'/'.join(title(plot)): figure}
