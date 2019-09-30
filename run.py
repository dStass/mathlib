from library.vector import Vector
from library.plotter import Plotter
import random
import math

points = 50
lo, hi = -5,5

def f(x):
  if x == 0:
    return 0
  return abs(x**3)

# vs = []
# for i in range(points):
#   x = random.uniform(lo,hi)
#   v = Vector([x, f(x)])
#   vs.append(v)

# Plotter().plot(vs)


# p = Plotter()
# for c in range(2):
#   vs = []
#   for i in range(points):
#     x = random.uniform(lo,hi)
#     y = random.uniform(lo,hi)
#     v = Vector([x, y])
#     vs.append(v)

#   colour = 'k'
#   if c == 0:
#     colour = 'k'
#   else:
#     colour = 'r'
#   p.plot_vectors_from_origin(vs, colour)

p = Plotter()
v1 = Vector([0,1])
v2 = Vector([0,2])
v3 = Vector([1,1]) * 3
v4 = v3.unit()

p.add_arrow_vector(v1)
p.add_arrow_vector(v2)
p.add_arrow_vector(v3, 'g')
p.add_arrow_vector(v4, 'r')

print(Vector().dtheta(v1,v2))


p.save()
