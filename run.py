import os
import math
import random

from library.function import Function
from library.vector import Vector
from library.plotter import Plotter

VECS = 5

p = Plotter()

p.add_arrow_vector(v1)
p.add_arrow_vector(v2)
p.add_arrow_vector(v3, 'g')
p.add_arrow_vector(v4, 'r')
colours = ['b', 'g', 'k', 'y', 'm', 'c']
for i, v in enumerate(vs):
  colour = 'k'
    colour = 'r'


p.draw()
p.save()
os.system('cmd.exe /C start plot.png')