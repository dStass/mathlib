from multipledispatch import dispatch
import numpy as np
import matplotlib.pyplot as plt
import math

from library.vector import Vector
import random

class Plotter:
  PBUFFER = 0.1  # plot buffer
  xrange = [0,0]
  yrange = [0,0]
  x_buf = 0
  y_buf = 0

  arrows = []

  def Plotter(self):
    pass

  def plot(self, points, show = False):
    extracted = self.extract_vectors_and_ranges(points)
    xs = extracted[0]
    ys = extracted[1]
    range_x = extracted[2]
    range_y = extracted[3]

    self.update_xrange(range_x)
    self.update_x_buf()
    self.update_yrange(range_y)
    self.update_y_buf()

    # resize canvas
    self.resize()

    # draw origin axes
    self.draw_axis()
    
    plt.plot(xs, ys, 'ro', marker='.')

    if show: plt.show()

  def plot_data_sets(self, point_sets, show = False):
    '''
    Plot a range of points
    '''

    colours = ['r','b', 'k', 'c', 'm', 'y', 'g', 'w']
    # random.shuffle(colours)

    all_xs = []
    all_ys = []

    for points in point_sets:
      extracted = self.extract_vectors_and_ranges(points)
      xs = extracted[0]
      ys = extracted[1]
      # range_x = extracted[2]
      # range_y = extracted[3]

      # self.update_xrange(range_x)
      # self.update_x_buf()
      # self.update_yrange(range_y)
      # self.update_y_buf()

      # # resize canvas
      # self.resize()

      # # draw origin axes
      # self.draw_axis()

      all_xs.append(xs)
      all_ys.append(ys)
    

    # draw points
    colour_index = 0
    for xs, ys in zip(all_xs, all_ys):
      plt.scatter(xs, ys, color=colours[colour_index], linewidths=0.4)
      colour_index += 1
      colour_index %= len(colours)

    if show: plt.show()

  # def plot_vectors_from_origin(self, vectors, colour = 'k'):
  #   extracted = self.extract_vectors_and_ranges(vectors)
  #   xs = extracted[0]
  #   ys = extracted[1]
  #   range_x = extracted[2]
  #   range_y = extracted[3]

  #   self.update_xrange(range_x)
  #   self.update_x_buf()
  #   self.update_yrange(range_y)
  #   self.update_y_buf()

  #   arrow_buff = self.get_arrow_buffer()
    
  #   # resize canvas
  #   self.resize()

  #   # draw origin axes
  #   self.draw_axis()

  #   for i in range(len(xs)):
  #     xi = xs[i]
  #     yi = ys[i]
  #     mag = math.sqrt(xi**2 + yi**2)
  #     # plt.arrow(0, 0, xs[i], ys[i],  head_width= self.PBUFFER * arrow_buff * 1.5,
  #     #                               head_length=self.PBUFFER * arrow_buff * 2.5,
  #     #                               color=colour)
  #     plt.arrow(0, 0, xi, yi, head_width= mag*0.005,
  #                             head_length= mag * 0.001,
  #                             color=colour)



  def extract_vectors_and_ranges(self, vectors):
    range_x = [0,0]
    range_y = [0,0]

    xs = []
    ys = []
    for i in range(len(vectors)):
      x = vectors[i][0]
      y = vectors[i][1]
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

  def update_xrange(self, range):
    if range[0] < self.xrange[0]:
      self.xrange[0] = range[0]
    if range[1] > self.xrange[1]:
      self.xrange[1] = range[1]
  
  def update_x_buf(self):
    self.x_buf = abs(self.xrange[1] - self.xrange[0]) * self.PBUFFER
  
  def update_yrange(self, range):
    if range[0] < self.yrange[0]:
      self.yrange[0] = range[0]
    if range[1] > self.yrange[1]:
      self.yrange[1] = range[1]

  def update_y_buf(self):
    self.y_buf = abs(self.yrange[1] - self.yrange[0]) * self.PBUFFER


  def resize(self):
    plt.xlim(self.xrange[0] - self.x_buf , self.xrange[1] + self.x_buf)
    plt.ylim(self.yrange[0] - self.y_buf , self.yrange[1] + self.y_buf)
  
  def draw_axis(self):
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

  def add_arrow_vector(self, vector, colour = 'k', from_vec = Vector([0,0])):
    self.update_size_if_required(vector)
    arrow_buff = self.get_arrow_buffer()
    self.arrows.append([from_vec, vector, colour])
    
  
  def draw(self):

    arrow_buf = self.get_arrow_buffer()
    for arrow in self.arrows:
      arrow_from = arrow[0]
      arrow_to = arrow[1]
      arrow_colour = arrow[2]
      
      arrow_from_x = arrow_from.x()
      arrow_from_y = arrow_from.y()

      arrow_to_x = arrow_to.x()
      arrow_to_y = arrow_to.y()



      plt.arrow(  arrow_from_x,
                  arrow_from_y,
                  arrow_to_x,
                  arrow_to_y,
                  head_width = arrow_buf*0.15,
                  head_length = arrow_buf*0.3,
                  color = arrow_colour,
                  length_includes_head = True)

  def get_arrow_buffer(self):
    return math.sqrt(self.x_buf**2 + self.y_buf**2)

  def update_size_if_required(self, vector):
    change = False
    x = vector.x()
    y = vector.y()
    if x < self.xrange[0]:
      change = True
      self.xrange[0] = x
      self.update_x_buf()
    if x > self.xrange[1]:
      change = True
      self.xrange[1] = x
      self.update_x_buf()

    if y < self.yrange[0]:
      change = True
      self.yrange[0] = y
      self.update_y_buf()
    
    if y > self.yrange[1]:
      change = True
      self.yrange[1] = y
      self.update_y_buf()

    if change:
      self.resize()  # TODO: FIX user warning

  def save(self, file_name = 'plot', path = ''):
    EXT = '.png'
    plt.savefig(file_name + EXT, format='png', dpi=1200)
