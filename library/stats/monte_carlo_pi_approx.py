'''
The area of a circle is defined as πr^2.
Estimate π to 3 decimal places using a Monte Carlo method.
Hint: The basic equation of a circle is x2 + y2 = r2.
'''
import random
import statistics
import math

TRIALS = 10
SAMPLE = 10

random.seed()
# randomly generate x and y from 0->1
# if x^2 + y^2 < 1, add to a count
def approx():
  pis = generate_pis()
  return statistics.mean(pis)

def generate_pis(trials = TRIALS, sample = SAMPLE):
  pis = []
  for _ in range(trials):
    count = 0
    for _ in range(sample):
      x = random.uniform(-1,1)
      y = random.uniform(-1,1)
      if x*x + y*y <= 1:
        count += 1
    count_average = count/sample
    area = count_average * 4 # since we are looking at area from -1 < x,y < 1, total area = 2^2 = 4
    pis.append(area)
  return pis


# print("pi=", approx())


