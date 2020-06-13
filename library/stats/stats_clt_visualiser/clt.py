'''
CLT
'''

import numpy as np
import matplotlib.pyplot as plt
import math


SAMPLE_SIZE = 10
POISSON_LAMBDA = 2000
NUM_GENERATE = 10000

# generate N items in distribution
def generate_n_items_from_dist(n, dist_name):
  if dist_name == "poisson":
    return generate_poisson(POISSON_LAMBDA, SAMPLE_SIZE)

# poisson lambda = 1
# P(X) = labda^(-x)*e
def generate_poisson(lmbd, n):
  to_return = np.random.poisson(lam=lmbd, size=n)
  return to_return

# get mean distribution
# get mean in trials and how many times the same mean occurred
# returns a list of 2 lists (means and its occurence by index)
def get_mean_distribution(n, dist_name):
  m = generate_n_items_from_dist(n, dist_name)
  return val_and_count(m)


def val_and_count(values):
  occurrences = {}
  for v in values:
    occurrence[v] = occurrences.get(v, 0) + 1

  to_return = []
  val = []
  count = []

  for o in occurrences:
    val.append(float(v))
    count.append(float(occurrences[v]))

  to_return.append(val)
  to_return.append(count)
  return to_return

# Math functions
# mean of a list rounded to 2 dp
def mean(l):
  summed = float(sum(l))
  total = float(len(l))
  m = summed / total
  # print(summed, total)
  return m


def variance(l, sample_mean):
  var = 0
  for m in l:
    var += float((m - sample_mean)**2)/float(len(l))
  return var

def standev(l, sample_mean):
  var = variance(l, sample_mean)
  sd = math.sqrt(var)
  return sd

# Visualisation
def bar_plot(x,y):
  plt.bar(x, y)
  plt.savefig('plot.png')

# main:
mean_distribution = get_mean_distribution(NUM_GENERATE, "poisson")

means = mean_distribution[0]
means_times_occurred = mean_distribution[1]
# means_times_probability = [p / float(SAMPLE_SIZE) for p in means_times_occurred]

means_sample_mean = mean(means)
means_se = standev(means, means_sample_mean)

z_means = [round((m - means_sample_mean)/means_se, 2) for m in means]
p_z_means = val_and_count(z_means)[1]
# print(means_times_occurred)

# bar_plot(means, means_times_occurred)
# bar_plot(means, means_times_probability)
bar_plot(z_means, p_z_means)
