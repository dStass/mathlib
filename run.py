from library.vector import Vector
from library.plotter import Plotter
import random
import math

points = 100
lo, hi = -20.0, 20.0

def f(x):
  if x == 0:
    return 0
  return abs(1/x)

# vs = []
# for i in range(points):
#   x = random.uniform(lo,hi)
#   v = Vector([x, f(x)])
#   vs.append(v)

# Plotter().plot(vs)

vs = []
for i in range(points):
  x = random.uniform(lo,hi)
  y = random.uniform(lo,hi)
  v = Vector([x, y])
  vs.append(v)

Plotter().plot_vectors_from_origin(vs)
