import math
import random

import numpy as np
from scipy import stats

class Distribution:
  OUTLIER_DEVIATION_CUTOFF = 0.5
  MAX_INT = 2 ** 31 - 1
  EPS = 0.000005

  def generate_with_outliers(self, arg0 = 1, arg1 = 0, N = 10):
    return self.generate(arg0, arg1, N)

  def evaluate_at(self, x, pdf_i):
    return None

  def get_random_variate(self, pdf_i):
    return None

  def generate_pdf_set(self, weights):
    return None

  def num_deviations_from_average(self, value, pdf_i):
    return None

  def generate_weighted_pdfs(self, pdf_set, N = 10):
    '''
    pdf_set: list of tuples of length 3.
    where the ith tuple contains (weight_i, mu_i, sigma_i)
    condition: sum of all weight_i = 1
    '''
    dataset = []
    for pdf_i in pdf_set:
      weight_i = pdf_i[0]
      num_values_pdf_i = int(weight_i * N)
      for i in range(num_values_pdf_i):
        random_variate = self.get_random_variate(pdf_i, self.OUTLIER_DEVIATION_CUTOFF)
        dataset.append(random_variate)

    # ensure dataset has length N
    while len(dataset) > N: dataset.pop()
    while len(dataset) < N: dataset.append(self.get_random_variate(pdf_i, self.OUTLIER_DEVIATION_CUTOFF))

    return dataset


  def get_combined_weighted_pdf_plot_points(self, _from, _to, _inc, pdf_set):
    '''
    pdf_set: list of lists of length 3.
    where the ith list contains (mu_i, sigma_i, weight_i)
    condition: sum of all weight_i = 1
    '''
    points = []

    for x in np.arange(_from, int(_to + _inc/2.0), _inc):
      y = 0
      for pdf_i in pdf_set:
        y += self.evaluate_at(x, pdf_i)
      if y > self.EPS: points.append((x,y))
    return points


  def generate_data_with_outliers(self, mean = 10, outlier_amount = 0.05, outlier_left_skew = 0.5, outlier_skew_random = False, inc = 100, N = 100):
    '''
    generate a dataset with outliers
    '''

    if outlier_skew_random: outlier_left_skew = random.random()

    standard_set_fraction = 1 - outlier_amount
    outlier_left_fraction = outlier_left_skew * outlier_amount
    outlier_right_fraction = outlier_amount - outlier_left_fraction

    weights = [standard_set_fraction, outlier_left_fraction, outlier_right_fraction]
    pdf_set = self.generate_pdf_set(weights, mean)

    data = self.generate_weighted_pdfs(pdf_set, N)
    # data = [(d, 0) for d in data]
    # plot_points = self.get_combined_weighted_pdf_plot_points(pdf_set[1][1]-10, pdf_set[2][1]+10, 1/inc, pdf_set)

    return data, weights


  # def generate_random(self, size = 1000):
  #   # size_total = random.randint(500, 5000)
  #   size_total = size
  #   num_outliers = random.randint(1, int(0.01*size_total))

  #   median = random.randint(500,1000)
  #   err = (random.randint(500,2000)/10000)*median # 5% error
  #   outlier_err = (random.randint(5000, 8000)/10000)*median

  #   num_lower_outliers = int(random.random() * num_outliers)
  #   num_lower_outliers = int(0.999 * num_outliers)
  #   num_higher_outliers = num_outliers - num_lower_outliers

  #   # generate standard dataset without any outliers
  #   errs = err * np.random.rand(size_total-num_outliers) * np.random.choice((-1, 1), size_total-num_outliers)
  #   data = median + errs
  #   correct_data = np.sort(data).tolist()

  #   # generate outliers on bottom end of dataset
  #   lower_errs = outlier_err * np.random.rand(num_lower_outliers)
  #   lower_outliers = median - err - lower_errs

  #   # generate outliers on upper end of dataset
  #   upper_errs = outlier_err * np.random.rand(num_higher_outliers)
  #   upper_outliers = median + err + upper_errs

  #   data = np.concatenate((data, lower_outliers, upper_outliers))
  #   outlier_data = np.sort(data).tolist()

  #   np.random.shuffle(data)

  #   return_data = {}
  #   return_data['num_values'] = size_total
  #   return_data['num_outliers'] = num_outliers
  #   return_data['data'] = data
  #   return_data['median'] = median
  #   return_data['correct_range'] = [correct_data[0], correct_data[-1]]
  #   return_data['outlier_range'] = [outlier_data[0], outlier_data[-1]]

  #   return return_data


class ExponentialDistribution(Distribution):
  def evaluate_at(self, x, pdf_i):
    weight_i = pdf_i[0]
    lambda_i = pdf_i[1]
    if x < 0: y = 0
    else:
      try: y = weight_i * lambda_i * (math.exp(-lambda_i * x))
      except: y = self.MAX_INT
    return y

  def get_random_variate(self, pdf_i):
    lambda_i = pdf_i[1]
    random_variate = random.expovariate(lambda_i)
    return random_variate

  # https://math.stackexchange.com/questions/741118/standard-deviation-with-exponential-distribution


class GammaDistribution(Distribution):
  def evaluate_at(self, x, pdf_i):
    weight_i = pdf_i[0]
    alpha_i = pdf_i[1]
    theta_i = pdf_i[2]
    y = weight_i * stats.gamma.pdf(x, a = alpha_i, scale = theta_i)
    return y

  def get_random_variate(self, pdf_i):
    alpha_i = pdf_i[1]
    theta_i = pdf_i[2]
    random_variate = random.gammavariate(alpha_i, theta_i)
    return random_variate 

class UniformDistribution(Distribution):
  def evaluate_at(self, x, pdf_i):
    weight_i = pdf_i[0]
    lower_i = pdf_i[1]
    higher_i = pdf_i[2]
    if lower_i <= x <= higher_i: y = weight_i * 1/(higher_i - lower_i)
    else: y = 0
    return y

  def get_random_variate(self, pdf_i):
    lower_i = pdf_i[1]
    higher_i = pdf_i[2]
    random_variate = lower_i + random.random() * (higher_i - lower_i)
    return random_variate 


class NormalDistribution(Distribution):
  def evaluate_at(self, x, pdf_i):
    weight_i = pdf_i[0]
    mu_i = pdf_i[1]
    sigma_i = pdf_i[2]
    y = weight_i * (1/(sigma_i * math.sqrt(2*math.pi))) * math.exp(-(((x-mu_i)**2)/(2*sigma_i**2)))
    return y

  def get_random_variate(self, pdf_i, max_deviation = None):
    mu_i = pdf_i[1]
    sigma_i = pdf_i[2]
    if not max_deviation: return random.normalvariate(mu_i, sigma_i)
    while True:
      random_variate = random.normalvariate(mu_i, sigma_i)
      if self.within_acceptable_deviation(random_variate, pdf_i): return random_variate
  
  def generate_pdf_set(self, weights, centre):
    standard_set = (weights[0], centre, random.uniform(0.5, 2))
    left_set = (weights[1], centre - centre * standard_set[2] * random.uniform(1, 3), random.uniform(0.5, 2))
    right_set = (weights[2], centre + centre * standard_set[2] * random.uniform(1, 3), random.uniform(0.5, 2))
    pdf_set = [standard_set, left_set, right_set]

    if right_set[1] < left_set[1]: pdf_set = [standard_set, right_set, left_set]

    return pdf_set
  
  def within_acceptable_deviation(self, value, pdf_i):
    mu_i = pdf_i[1]
    sigma_i = pdf_i[2]
    num_deviations = abs(value - mu_i) / sigma_i
    if num_deviations < self.OUTLIER_DEVIATION_CUTOFF: return True
    else: return False