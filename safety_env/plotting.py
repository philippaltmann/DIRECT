import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import plotly.graph_objects as go

def heatmap_2D(data, vmin=0, vmax=1):
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
  if vmin!=0: vmin = vmin or min(flat) 
  if vmax!=0: vmax = vmax or max(flat)
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


def heatmap_3D(data, vmin=0, vmax=1, show_agent=False):
  rows,cols  = data.shape[0:2] # extract iteration data and arguments
  
  # Value range and coloration helpers
  fltd = [np.mean(v) for row in data for col in row for v in col if v is not None and not None in v]
  vmin = min(fltd) if vmin is None else vmin; vmax = max(fltd) if vmax is None else vmax
  pct = lambda v: (np.mean(v)-vmin)/(vmax-vmin); opc = lambda v: 1-np.var(v)/(vmax-vmin)**2*2 # alt: 1-np.std(test)/(10-0)*2
  lookup = {'f':'hsl({:0.0f},{:0.0f}%,80%)', 'g':'hsl(140,100%,50%)', 'l': 'hsl(0,100%,50%)', 'w': 'rgba(192,192,192,0.5)'}
  color = lambda v: lookup[v] if type(v) == str and v in ['a','g','l','w'] else lookup['f'].format(pct(v)*140, opc(v)*100)
  # lookup = {'f':'hsla({:0.0f},80%,40%,{:1.2f})', 'g':'hsl(140,100%,60%)', 'l': 'hsl(0,100%,60%)', 'w': 'rgba(192,192,192,0.5)'}
  # color = lambda v: lookup[v] if type(v) == str and v in ['g','l','w'] else lookup['f'].format(pct(v)*120, opc(v)) #140
  t = lambda c,r: 'w' if r in [0,rows-1] or c in [0,cols-1] else 'g' if r in [1,rows-2] and c in [1,cols-2] else 'l'
  
  # Reference Point constructors for triangulations
  ct = lambda c,r,h=0: (c,r,h) # Helper points at c(ollumn), r(ow), h(ight) for w(idth)
  ul = lambda c,r,h=0,w=0.5: (c-w, r-w, h); ur = lambda c,r,h=0,w=0.5: (c+w, r-w, h)
  dr = lambda c,r,h=0,w=0.5: (c+w, r+w, h); dl = lambda c,r,h=0,w=0.5: (c-w, r+w, h)
  points = lambda c,r,v: [ct(c,r,0),ul(c,r,0),ur(c,r,0),dr(c,r,0),dl(c,r,0),ul(c,r,1),ur(c,r,1),dr(c,r,1),dl(c,r,1)]

  #  Reference Point constructors for positioning text
  uc = lambda c,r,w=0.3: (c, r-w, 0.1); rc = lambda c,r,w=0.3: (c+w, r, 0.1)
  dc = lambda c,r,w=0.3: (c, r+w, 0.1); lc = lambda c,r,w=0.3: (c-w, r, 0.1)
  labels = lambda c,r: [uc(c,r),lc(c,r),dc(c,r),rc(c,r)]
  txt = lambda c,r,val: [(*p, f'{np.mean(v):.2f}', 'black') for p,v in zip(labels(c,r), val)] if not None in val.flat else []

  # Indices of points to build triangle (up, right, down, left)
  _adapt = lambda c,r,t: tuple(map(lambda v: v+9*(r*cols+c),t))
  triang = lambda c,r,i: _adapt(c,r, [(0,1,2), (0,4,1), (0,3,4), (0,2,3)][i]) #urdl(0,1,2), (0,2,3), (0,3,4), (0,4,1)
  square = lambda c,r: [_adapt(c,r,v) for v in [(1,2,3), (1,3,4), (6,7,8), (5,6,8), (1,2,6), (2,5,6), (2,3,6), (3,6,7), (3,4,8), (3,7,8), (1,4,8), (1,5,8)]]
  field = lambda c,r,val: [] if None in val.flat else [(*triang(c,r,i), color(v)) for i,v in enumerate(val)] 
  cube = lambda c,r,val: [(*f, color(t(c,r))) for f in square(c,r)] if None in val.flat else []
  agent = lambda c,r,val: [(*f, 'hsl(210,100%,50%)') for f in square(c,r)] if show_agent and c==cols-2 and r==1 else []

  process = lambda *f: np.array([result for fn in f for r, row in enumerate(data) for c,val in enumerate(reversed(row)) for result in fn(c,r,val)]).T.tolist()

  p = dict(zip(['x','y','z'], process(points))); f = dict(zip(['i','j','k','facecolor'], process(field, cube, agent)))
  l = dict(zip(['x','y','z','text','textfont'], process(txt))); l['textfont'] = {'color': l['textfont'], 'size':8} 

  mesh=go.Mesh3d(**p, **f) 
  text=go.Scatter3d(**l, mode='text', textposition='middle center') 

  axis = {'showbackground':False, 'tickmode': 'linear', 'range':[-0.5,max(data.shape)-0.5], 'visible': False} 
  scene = {'xaxis':axis, 'yaxis':axis, 'zaxis':axis, 'camera': {'eye': {'x':0, 'y':0.125, 'z':0.6}}} 
  layout = go.Layout(margin=dict(l=8,r=8,t=8,b=8), width=900, height=600, scene=scene) #title=title
  return go.Figure(data=[mesh,text], layout=layout)
