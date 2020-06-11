import random

class ExponentialDistribution:
  def generate(self, _lambda = 1, N = 10):
    dataset = []
    for _ in range(N):
      dataset.append(random.expovariate(_lambda))
    return dataset


