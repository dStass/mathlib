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


  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # #                   VECTOR PROPERTIES                   # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

  # return acute angle between two vectors
  # in degrees
  def dtheta(self, v1, v2):
    if v1.__dim != v2.__dim:
      return None

    dot = v1 * v2
    mag1 = v1.magnitude()
    mag2 = v2.magnitude()
    if mag1 == 0 and mag2 == 0:
      return 0
    
    dot_over_mags = dot/(mag1 * mag2)
    rounded_dom = round(dot_over_mags, 10) # ie 1.000000
    if rounded_dom == 1:
      return 0
    if rounded_dom == -1:
      return 180
  
    angle_in_radians = math.acos(dot_over_mags)
    angle_in_degrees = math.degrees(angle_in_radians)
    if angle_in_degrees > 180:
      angle_in_degrees = 360 - angle_in_degrees
    return angle_in_degrees


  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # #                         HELPER                        # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  def copy(self):
    c = Vector((self.__coordinates).copy())
    return c

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # #                       OPERATORS                       # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #                       MULTILPICATION                      #
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

  @dispatch(object)
  def __mul__(self, vector):
    if self.__dim != vector.__dim:
      return None
    
    prod = 0
    for i in range(self.__dim):
      prod += (self.__coordinates[i] * vector.__coordinates[i])
    return prod


  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #                          DIVISION                         #
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
  #                         ADDITION                          #
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