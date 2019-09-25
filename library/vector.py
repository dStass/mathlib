from multipledispatch import dispatch

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
  
  # Operators:

  # Multiply *=
  @dispatch(int)
  def __imul__(self, scalar):
    print(self)
    for i in range(self.__dim):
      self.__coordinates[i] *= scalar
    return self

  @dispatch(float)
  def __imul__(self, scalar):
    for i in range(self.__dim):
      self.__coordinates[i] *= scalar
    return self