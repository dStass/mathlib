import os
import math
import random

from library.function import Function
from library.vector import Vector
from library.plotter import Plotter

VECS = 5

p = Plotter()

vs = []
for i in range(VECS):
  if i == VECS-1:
    v = Vector([0,0])
    for j in range(VECS-1):
      vj = vs[j]
      v += vj
    vs.append(v/VECS)
  else:
    vs.append(Vector([random.uniform(0,5), random.uniform(-5,5)]))


colours = ['b', 'g', 'k', 'y', 'm', 'c']
for i, v in enumerate(vs):
  colour = 'k'
  if i == len(vs) - 1:
    colour = 'r'
  p.add_arrow_vector(v, colour)


p.draw()
p.save()
os.system('cmd.exe /C start plot.png')