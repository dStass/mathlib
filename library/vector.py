from multipledispatch import dispatch
import math

class Vector:
  def __init__(self, values = None):
    self.__dim = 0
    self.__coordinates = []
    if values:
      self.__dim = len(values)
      self.__coordinates = values
  
  def __str__(self):
    to_return = '['
    for i in range(self.__dim):
      to_return += str(self.__coordinates[i])
      if i != self.__dim - 1:
        to_return += ', '
    to_return += ']'
    return to_return

  def x(self):
    return self.__coordinates[0]

  def y(self):
    return self.__coordinates[1]
  
  def z(self):
    return self.__coordinates[2]

  def magnitude(self):
    total = 0
    for c in self.__coordinates:
      total += c**2
    return math.sqrt(total)

  # return unit vector of self
  def unit(self):
    v = self.copy()
    mag = self.magnitude()
    for i in range(len(v.__coordinates)):
      v.__coordinates[i] /= mag
    return v

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #                           HELPER                         #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  
  def copy(self):
    c = Vector((self.__coordinates).copy())
    return c


  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #                         OPERATORS                         #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #                          MULTILPY                         #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  # *= scalar:int
  @dispatch(int)
  def __imul__(self, scalar):
    for i in range(self.__dim):
      self.__coordinates[i] *= scalar
    return self

  # *= scalar:float
  @dispatch(float)
  def __imul__(self, scalar):
    for i in range(self.__dim):
      self.__coordinates[i] *= scalar
    return self

  # * scalar:int
  @dispatch(int)
  def __mul__(self, scalar):
    new_copy = self.copy()
    for i in range(new_copy.__dim):
      new_copy.__coordinates[i] *= scalar
    return new_copy

  # * scalar:float
  @dispatch(float)
  def __mul__(self, scalar):
    new_copy = self.copy()
    for i in range(new_copy.__dim):
      new_copy.__coordinates[i] *= scalar
    return new_copy


  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #                           DIVIDE                          #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  # /= scalar:int
  @dispatch(int)
  def __itruediv__(self, scalar):
    if scalar != 0:
      self *= 1/float(scalar)
    return self

  # /= scalar:float
  @dispatch(float)
  def __itruediv__(self, scalar):
    self = self.copy()
    if scalar != 0:
      self *= 1/float(scalar)
    return self

  # / scalar:int
  @dispatch(int)
  def __truediv__(self, scalar):
    new_copy = self.copy()
    if scalar != 0:
      new_copy *= 1/float(scalar)
    return new_copy

  # / scalar:float
  @dispatch(float)
  def __truediv__(self, scalar):
    new_copy = self.copy()
    if scalar != 0:
      new_copy *= 1/float(scalar)
    return new_copy

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #                            PLUS                           #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  # += object:Vector
  def __iadd__(self, vec):
    if type(vec) is Vector and vec.__dim == self.__dim:
      for i in range(self.__dim):
        self.__coordinates[i] += vec.__coordinates[i]
      return self
    else:
      return None

  # + object:Vector
  def __add__(self, vec):
    if type(vec) is Vector and vec.__dim == self.__dim:
      new_copy = self.copy()
      for i in range(self.__dim):
        new_copy.__coordinates[i] += vec.__coordinates[i]
      return new_copy
    else:
      return None


  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #                        SUBTRACTION                        #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  # += object:Vector
  def __isub__(self, vec):
    if type(vec) is Vector and vec.__dim == self.__dim:
      for i in range(self.__dim):
        self.__coordinates[i] -= vec.__coordinates[i]
      return self
    else:
      return None

  # + object:Vector
  def __sub__(self, vec):
    if type(vec) is Vector and vec.__dim == self.__dim:
      new_copy = self.copy()
      for i in range(self.__dim):
        new_copy.__coordinates[i] -= vec.__coordinates[i]
      return new_copy
    else:
      return None