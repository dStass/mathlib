from monte_carlo_pi_approx import generate_pis
import statistics
import math
from collections import deque
import matplotlib.pyplot as plt

def analyse():

  zsc = 0.1
  better = 0
  worse = 0
  min_dif = 100
  min_at = 0
  results = deque()

  while zsc <= 1.5:
    pis = generate_pis()
    pi_mean = statistics.mean(pis)
    pi_stdev = statistics.stdev(pis)
    old_mean = pi_mean
    old_med = statistics.median(pis)
    old_dif = abs(pi_mean / math.pi - 1)
    filtered_pis = []
    for p in pis:
      z = (p - pi_mean) / pi_stdev

      if abs(z) <= zsc: 
        filtered_pis.append(p)
    new_mean = 0
    if len(filtered_pis) > 0:
      new_mean = statistics.mean(filtered_pis)
    new_dif = abs(new_mean / math.pi - 1)
    if new_mean > 0:
      results.append(new_mean/math.pi)
    if abs(new_mean - math.pi) < min_dif:
      min_dif = abs(new_mean - math.pi)
      min_at = zsc

    if old_dif > new_dif:
      better += 1
    elif old_dif == new_dif:
      pass
    else:
      worse += 1
    zsc += 0.01
    # print("Completed ... ", zsc, sep = '')
  
  plt.plot(results)
  plt.savefig('plot.png')
  return round(min_at, 2)
  # print("better = ", better, ", worse = ", worse, ", oldmean = ", old_mean, ", old_med = ", old_med, sep = '')


test_range = 10
zscores = 0
for i in range(test_range):
  z = analyse()
  zscores += z
  print("BEST ZSC = ", z)
print("average =", zscores/test_range)
# data = deque()
# for i in range(100):
#   data.append(analyse())
#   print("Completed: ", i)


# plt.hist(data)
# plt.savefig('plot.png')
