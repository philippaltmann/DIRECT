import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

def triangle_heatmap(data, min=0, max=1):
  # TODO: automatic min max scaling 
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
  # plot = lambda v,c,r: ax.tripcolor(square(c,r), v, cmap='Spectral',vmin=-51, vmax=49, ec='w') #vmin=0, vmax=1, RdYlGn
  # text = lambda v,c,r: [ax.text(x,y, f'{v:.0f}', color='k' if -10 < v < 10 else 'w', ha='center', va='center') for x,y,v in zip(*labels(c,r).T, v)]
  
  plot = lambda v,c,r: ax.tripcolor(square(c,r), v, cmap='Spectral', vmin=min, vmax=max, ec='w') #, RdYlGn
  center = min + (max - min) / 2; width = abs(min) + abs(max)
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
