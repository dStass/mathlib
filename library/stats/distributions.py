import math
import random

import numpy as np

class Distribution:
  def generate_with_outliers(self, arg0 = 1, arg1 = 0, N = 10):
    return self.generate(arg0, arg1, N)

  def generate_random(self, size = 1000):
    # size_total = random.randint(500, 5000)
    size_total = size
    num_outliers = random.randint(1, int(0.01*size_total))

    median = random.randint(500,1000)
    err = (random.randint(500,2000)/10000)*median # 5% error
    outlier_err = (random.randint(5000, 8000)/10000)*median

    num_lower_outliers = int(random.random() * num_outliers)
    num_lower_outliers = int(0.999 * num_outliers)
    num_higher_outliers = num_outliers - num_lower_outliers

    # generate standard dataset without any outliers
    errs = err * np.random.rand(size_total-num_outliers) * np.random.choice((-1, 1), size_total-num_outliers)
    data = median + errs
    correct_data = np.sort(data).tolist()

    # generate outliers on bottom end of dataset
    lower_errs = outlier_err * np.random.rand(num_lower_outliers)
    lower_outliers = median - err - lower_errs

    # generate outliers on upper end of dataset
    upper_errs = outlier_err * np.random.rand(num_higher_outliers)
    upper_outliers = median + err + upper_errs

    data = np.concatenate((data, lower_outliers, upper_outliers))
    outlier_data = np.sort(data).tolist()

    np.random.shuffle(data)

    return_data = {}
    return_data['num_values'] = size_total
    return_data['num_outliers'] = num_outliers
    return_data['data'] = data
    return_data['median'] = median
    return_data['correct_range'] = [correct_data[0], correct_data[-1]]
    return_data['outlier_range'] = [outlier_data[0], outlier_data[-1]]

    return return_data


class ExponentialDistribution(Distribution):
  def generate(self, _placeholder = 0, _lambda = 1, N = 10):
    dataset = []
    for _ in range(N):
      dataset.append(random.expovariate(_lambda))
    return dataset


class GammaDistribution(Distribution):
  def generate(self, _alpha = 1, _beta = 1, N = 10):
    dataset = []
    for _ in range(N):
      dataset.append(random.gammavariate(_alpha, _beta))
    return dataset


class UniformDistribution(Distribution):
  def generate(self, _lower = 0, _higher = 1, N = 10):
    dataset = []
    for _ in range(N):
      dataset.append(_lower + random.random() * (_higher - _lower))
    return dataset

  def get_plot_points(self, _from, _to, _inc):
    points = []
    y = int((_to + _inc/2.0 - _from) / _inc)
    for x in np.arange(_from, int(_to + _inc/2.0), _inc):
      points.append((x, y))
    return points


class NormalDistribution(Distribution):
  def __init__(self, _mu = 0, _sigma = 1):
    self._mu = _mu
    self._sigma = _sigma


  def generate(self, _mu = 1, _sigma = 1, N = 10):
    dataset = []
    for _ in range(N):
      dataset.append(random.normalvariate(_mu, _sigma))
    return dataset

  def generate_weighted_pdfs(self, pdf_set, N = 10):
    '''
    pdf_set: list of tuples of length 3.
    where the ith tuple contains (mu_i, sigma_i, weight_i)
    condition: sum of all weight_i = 1
    '''
    dataset = []
    for pdf_i in pdf_set:
      mu_i = pdf_i[0]
      sigma_i = pdf_i[1]
      weight_i = pdf_i[2]

      num_values_pdf_i = int(weight_i * N)
      for i in range(num_values_pdf_i):
        dataset.append(random.normalvariate(mu_i, sigma_i))

    # ensure dataset has length N
    while len(dataset) > N: dataset.pop()
    while len(dataset) < N: dataset.append(random.normalvariate(mu_i, sigma_i))

    return dataset


  def get_plot_points(self, _from, _to, _inc, _mu, _sigma):
    if not _mu: _mu = self._mu
    if not _sigma: _sigma = self._sigma
    points = []
    for x in np.arange(_from, int(_to + _inc/2.0), _inc):
      y = (1/(_sigma * math.sqrt(2*math.pi))) * math.exp(-(((x-_mu)**2)/(2*_sigma**2)))
      points.append((x,y))
    return points


  def get_combined_weighted_pdf_plot_points(self, _from, _to, _inc, pdf_set):
    '''
    pdf_set: list of tuples of length 3.
    where the ith tuple contains (mu_i, sigma_i, weight_i)
    condition: sum of all weight_i = 1
    '''
    weights = [t[2] for t in pdf_set]
    points = []
    for x in np.arange(_from, int(_to + _inc/2.0), _inc):
      y = 0
      for pdf_i in pdf_set:
        mu_i = pdf_i[0]
        sigma_i = pdf_i[1]
        weight_i = pdf_i[2]
        y += weight_i * (1/(sigma_i * math.sqrt(2*math.pi))) * math.exp(-(((x-mu_i)**2)/(2*sigma_i**2)))
      points.append((x,y))
    return points