import numpy as np; import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import plotly.graph_objects as go


def triangle_heatmap(data, vmin=0, vmax=1):
  figure, ax = plt.subplots(figsize=(8,6))
  # f = plt.figure(figsize=fig_size)
  ct = lambda c,r: (c,r) # Helper points at c(ollumn), r(ow) for w(idth)
  ul = lambda c,r,w=0.5: (c-w, r-w); ur = lambda c,r,w=0.5: (c+w, r-w)
  dr = lambda c,r,w=0.5: (c+w, r+w); dl = lambda c,r,w=0.5: (c-w, r+w)
  
  # For positioning text
  uc = lambda c,r,w=0.3: (c, r-w); rc = lambda c,r,w=0.3: (c+w, r)
  dc = lambda c,r,w=0.3: (c, r+w); lc = lambda c,r,w=0.3: (c-w, r)

  # Indices of points to build triangle (up, right, down, left)
  triangles = [(0,1,2), (0,2,3), (0,3,4), (0,4,1)]
  labels = lambda c,r: np.array([uc(c,r),rc(c,r),dc(c,r),lc(c,r)])
  points = lambda c,r: np.array([ct(c,r),ul(c,r),ur(c,r),dr(c,r),dl(c,r)])
  square = lambda c,r: Triangulation(*points(c,r).T, triangles)
  
  flat = [v for row in data for col in row for v in col if v is not None]
  vmin = vmin or min(flat); vmax = vmax or max(flat)
  plot = lambda v,c,r: ax.tripcolor(square(c,r), v, cmap='Spectral', vmin=vmin, vmax=vmax, ec='w') #, RdYlGn
  center = vmin + (vmax - vmin) / 2; width = abs(vmin) + abs(vmax)
  lcolor = lambda v: 'k' if center - 0.3 * width < v < center + 0.3 * width else 'w'
  _txt = lambda v,c,r: ax.text(c,r, f'{v:.2f}', color=lcolor(v), ha='center', va='center')
  text = lambda v,c,r: [_txt(v,x,y) for x,y,v in zip(*labels(c,r).T, v)]

  # Add triangles & labels
  imgs = [plot(v,c,r) for r, row in enumerate(data) for c, v in enumerate(row) if not (None in v)]
  [text(v,c,r) for r, row in enumerate(data) for c, v in enumerate(row) if len(v) if not (None in v)]

  # Add legends and axes
  figure.colorbar(imgs[0], ax=ax); ax.invert_yaxis();
  ax.set_xticks(range(len(data[0]))); ax.set_yticks(range(len(data))); 
  ax.margins(x=0, y=0); ax.set_aspect('equal', 'box'); plt.tight_layout()
  return figure


# Helper functions to create scatters/graphs from experiment & metric
def plot_box(title, plot):
  box = lambda g: go.Box(name=g['label'], y=g['data'], marker_color=color(g['hue']), boxmean=True)  #boxmean='sd'
  return { 'layout': layout(title, False), 'data': [ box(g) for g in plot['graphs'] ] }

# reward_threshold = {'type':'line', 'x0':0, 'y0':0.945, 'x1':1000000, 'y1':0.945, 'line':{'color':'#424242', 'width':2, 'dash':'dash'}}

def plot_ci(title, plot):
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=data, **kwargs)
  getmean = lambda g: scatter(g['data'][0], name=g['label'], mode='lines', line={'color': color(g['hue']), 'smoothing': 1.0})
  getconf = lambda g: scatter(g['data'][1], fillcolor=color(g['hue'], True), fill='toself', line={'color': 'rgba(255,255,255,0)'}, showlegend=False)
  #

  return { 'layout': layout(title), 'data': [getconf(g) for g in plot['graphs']] + [getmean(g) for g in plot['graphs']] }


#                      Red         Orange     Green          Blue        Purple 
# color_lookup = {"PPO": 340, "A2C": 40, "SIL": 130, "DIRECT": 210, "DQN": 290 }
def color(hue, sec=False): return 'hsva({},{}%,{}%,{})'.format(hue, 90-sec*20, 80+sec*20, 1.0-sec*0.8)


def layout(title=None, legend=True): 
  return go.Layout( title=title, showlegend=legend, font=dict(size=24), 
    margin=dict(l=8, r=8, t=8+(72 * (title is not None)), b=8), width=1200+300*legend, height=600)


def generate_figures(plots, generator):
  title = lambda plot: f'{plot["metric"]} ({plot["title"]})'
  return { title(plot): go.Figure(**generator[plot['metric']](title(plot), plot)) for plot in plots }
