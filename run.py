from library.vector import Vector
from library.plotter import Plotter
import random
import math

points = 10000
lo, hi = -20.0, 20.0

def f(x):
  return x**2*math.sin(x)/x*math.cos(x)

vs = []
for i in range(points):
  x = random.uniform(lo,hi)
  v = Vector([x, f(x)])
  vs.append(v)

Plotter().plot(vs)

v1 = Vector([1,0])
v2 = Vector([0,1])

v3 = Vector([1,1])

v3 += (v2 * 2)
v3 -= Vector([3,1])
print(v1)
print(v2)
print(v3)
