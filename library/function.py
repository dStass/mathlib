import numpy as np
import math
from library.vector import Vector  # use absolute path

class Function:
  EQUATIONS = [
    'cubic',
    'square',
    'cos',
    'sin'
  ]

  def cubic(x):
    return float(x**3)

  def square(x):
    return float(x**2)
  
  def sin(x):
    return math.sin(x)

  def cos(x):
    return math.cos(x)


  def plot_points(self, fr=-5.0, to=5.0, values=50, name='cubic'):
    vs = []

    points = np.linspace(fr, to, values, endpoint=True)


    # default cubic function
    f = self.cubic

    # check if given name exists and if so, let f be function
    if name in self.EQUATIONS:
      f = getattr(Function, name)  # returns the function 
      
    for i in points:
      vs.append(Vector([i, f(i)]))
    return vs