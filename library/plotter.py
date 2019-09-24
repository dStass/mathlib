import numpy as np
import matplotlib.pyplot as plt

from library.vector import Vector

class Plotter:
  PBUFFER = 0.1
  def plot(self, points):
    
    range_x = [0,0]
    range_y = [0,0]


    xs = []
    ys = []
    for i in range(len(points)):
      x = points[i].x()
      y = points[i].y()
      xs.append(x)
      ys.append(y)

      # set ranges
      if x < range_x[0]:
        range_x[0] = x
      if x > range_x[1]:
        range_x[1] = x

      if y < range_y[0]:
        range_y[0] = y
      if y > range_y[1]:
        range_y[1] = y

    x_buf = abs(range_x[1] - range_x[0]) * self.PBUFFER
    y_buf = abs(range_y[1] - range_y[0]) * self.PBUFFER

    # resize canvas
    plt.xlim(range_x[0] - x_buf , range_x[1] + x_buf)
    plt.ylim(range_y[0] - y_buf , range_y[1] + y_buf)

    # draw origin axes
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    
    plt.plot(xs, ys, 'ro', marker='.')

    
    plt.savefig('plot.png')