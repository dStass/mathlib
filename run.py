import os
import math
import random

from library.function import Function
from library.vector import Vector
from library.plotter import Plotter


# vs = Function().plot_points(fr=-10, to=math.pi*20, name='sin', values=5000)

# p = Plotter()
# p.plot(vs)


# p = Plotter()
# for c in range(2):
#   vs = Function().plot_points(fr=-10, to=math.pi*20, name='sin', values=50)

#   colour = 'k'
#   if c == 0:
#     colour = 'k'
#   else:
#     colour = 'r'
#   p.plot_vectors_from_origin(vs, colour)



p = Plotter()
v1 = Vector([1,1])
v2 = Vector([0,1]) + v1 * 10
v3 = v1 * 30
v4 = v3.unit()

p.add_arrow_vector(v1)
p.add_arrow_vector(v2)
p.add_arrow_vector(v3, 'g')
p.add_arrow_vector(v4, 'r')

# print(Vector().dtheta(v1,v2))

p.draw()
p.save()
os.system('cmd.exe /C start plot.png')