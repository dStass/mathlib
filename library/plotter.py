from multipledispatch import dispatch
import numpy as np
import matplotlib.pyplot as plt

from library.vector import Vector

class Plotter:
  PBUFFER = 0.1  # plot buffer
  def plot(self, points):
    extracted = self.extract_vectors_and_ranges(points)
    xs = extracted[0]
    ys = extracted[1]
    range_x = extracted[2]
    range_y = extracted[3]

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

  def plot_vectors_from_origin(self, vectors):
    extracted = self.extract_vectors_and_ranges(vectors)
    xs = extracted[0]
    ys = extracted[1]
    range_x = extracted[2]
    range_y = extracted[3]

    x_buf = abs(range_x[1] - range_x[0]) * self.PBUFFER
    y_buf = abs(range_y[1] - range_y[0]) * self.PBUFFER

    for i in range(len(xs)):
      plt.annotate(s='', xy=(xs[i],ys[i]), xytext=(0,0), arrowprops=dict(arrowstyle='->'))
    
     # resize canvas
    plt.xlim(range_x[0] - x_buf , range_x[1] + x_buf)
    plt.ylim(range_y[0] - y_buf , range_y[1] + y_buf)

    # draw origin axes
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    

    plt.savefig('plot.png')


  # return [[xlo, xhi], [ylo, yhi]]
  def extract_vectors_and_ranges(self, vectors):
    range_x = [0,0]
    range_y = [0,0]

    xs = []
    ys = []
    for i in range(len(vectors)):
      x = vectors[i].x()
      y = vectors[i].y()
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
    return [xs, ys, range_x, range_y]
    


  @dispatch(int, int)
  def add(self, x, y):
    print(x+y)
  
  @dispatch(str, str)
  def add(self, x, y):
    print("STR")
