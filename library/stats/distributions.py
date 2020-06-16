import math
import random

import numpy as np
from scipy import stats

class Distribution:
  OUTLIER_DEVIATION_CUTOFF = 3
  DRAW_DEVIATION_AMOUNT = 5
  MAX_INT = 2 ** 31 - 1
  EPS = 0.00000000000000000005

  OUTLIER_DEVIATION_RANGE = (0.5, 25)
  NON_OUTLIER_DEVIATION_RANGE = (0.5, 25)

  # s.d. other distributions should be from non-outlier set
  MIN_DEVIATION_FROM_TRUE_SET = 10

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

  def get_extreme_density_plot_points(self, pdf_i):
    return None
  
  def extract_range_int(self, _range):
    if len(_range) == 1: return _range[0]
    else: return random.randint(_range[0], _range[1])

  def extract_range(self, _range):
    '''
    If _range contains 1 item, we simply return it.
    Otherwise, we return return a random float between
    the first and second element of _range
    '''
    if len(_range) == 1: return _range[0]
    else: return random.uniform(_range[0], _range[1])

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


  def get_combined_weighted_pdf_plot_points(self, _inc, pdf_set):
    '''

    Normal case:
    pdf_set: list of lists of length 3.
    where the ith list contains (mu_i, sigma_i, weight_i)
    condition: sum of all weight_i = 1
    '''
    points = {}
    round_amount = len(str(_inc)[2:])
    for pdf_i in pdf_set:
      extremes = self.get_extreme_density_plot_points(pdf_i)
      lo, hi = extremes
      lo = round(lo, round_amount)
      hi = round(hi, round_amount)
      for x in np.arange(lo, int(hi + _inc/2.0), _inc):
        x = round(x, round_amount)
        y = self.evaluate_at(x, pdf_i)
        if x not in points: points[x] = 0
        points[x] += y
    
    to_return = [(x, points[x]) for x in points if points[x] > self.EPS]
    return to_return


  def generate_data_with_outliers(self, mean = [10], outlier_amount = [0.05], outlier_first_skew = None, inc = 100, N = [100], num_outlier_sources = [2], outlier_sides = []):
    '''
    generate a dataset with outliers

    num_outlier_sources: total number of different randomised outlier distributions
    all elements will sum to outlier_amount * N 
    '''
    mean = self.extract_range(mean)
    outlier_amount = self.extract_range(outlier_amount)
    print("outlierAMT = ", outlier_amount)

    
    N = self.extract_range(N)

    num_outlier_sources = self.extract_range_int(num_outlier_sources)

    if outlier_first_skew:
      outlier_first_skew = self.extract_range(outlier_first_skew)
      outlier_first_fraction = outlier_first_skew * outlier_amount
    else:
      outlier_first_fraction = random.uniform(0.0,1.0) * outlier_amount

    standard_set_fraction = 1 - outlier_amount
    weights = [standard_set_fraction, outlier_first_fraction]

    outlier_left_over = outlier_amount - outlier_first_fraction
    left_over_weights = np.random.random(size = num_outlier_sources - 1)
    left_over_weights /= left_over_weights.sum()
    left_over_weights *= outlier_left_over
    left_over_weights = left_over_weights.tolist()

    outlier_fraction_sum = outlier_first_fraction

    for index, weight in enumerate(left_over_weights):
      outlier_fraction_sum += weight
      weights.append(weight)
    
    weights[-1] += outlier_amount - outlier_fraction_sum

    pdf_set = self.generate_pdf_set(weights, mean, outlier_sides)
    print('weights', weights)

    data = self.generate_weighted_pdfs(pdf_set, N)

    # data = [(d, 0) for d in data]
    # plot_points = self.get_combined_weighted_pdf_plot_points(pdf_set[1][1]-10, pdf_set[2][1]+10, 1/inc, pdf_set)

    return data, pdf_set


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
  
  def generate_pdf_set(self, weights, centre, sides):
    pdf_set = []
    true_set = (weights[0], centre, random.uniform(self.NON_OUTLIER_DEVIATION_RANGE[0], self.NON_OUTLIER_DEVIATION_RANGE[1]))
    pdf_set.append(true_set)
    current_set_index = 0
    for weight in weights[1:]:
      if current_set_index < len(sides): positive_negative_factor = sides[current_set_index]
      else: positive_negative_factor = 1 if random.random() < 0.5 else -1
    
      # ensure generated outlier data exist atleast OUTLIER_DEVIATION_CUTOFF away from mean
      while True:
        curr_set = (weight, centre + positive_negative_factor * centre * true_set[2] * random.uniform(0.2,1), random.uniform(self.OUTLIER_DEVIATION_RANGE[0], self.OUTLIER_DEVIATION_RANGE[1]))
        
        if curr_set[1] < true_set[1]:
          max_deviation_from_true_set = (true_set[1] - abs(true_set[2]) * self.OUTLIER_DEVIATION_CUTOFF) - (curr_set[1] + abs(curr_set[1]) * self.OUTLIER_DEVIATION_CUTOFF)
        else:
          max_deviation_from_true_set = (curr_set[1] - abs(curr_set[2]) * self.OUTLIER_DEVIATION_CUTOFF) - (true_set[1] + abs(true_set[1]) * self.OUTLIER_DEVIATION_CUTOFF)
        
        if abs(max_deviation_from_true_set) > self.MIN_DEVIATION_FROM_TRUE_SET:
          pdf_set.append(curr_set)
          break
      current_set_index += 1



    return pdf_set
  
  def within_acceptable_deviation(self, value, pdf_i):
    mu_i = pdf_i[1]
    sigma_i = pdf_i[2]
    num_deviations = abs(value - mu_i) / sigma_i
    if num_deviations < self.OUTLIER_DEVIATION_CUTOFF: return True
    else: return False

  def get_extreme_density_plot_points(self, pdf_i):
    '''
    return extreme points we need to plot
    '''
    mu_i = pdf_i[1]
    sigma_i = pdf_i[2]

    min_range = mu_i - (self.DRAW_DEVIATION_AMOUNT + self.DRAW_DEVIATION_AMOUNT*sigma_i)
    max_range = mu_i + (self.DRAW_DEVIATION_AMOUNT + self.DRAW_DEVIATION_AMOUNT*sigma_i)
    return (min_range, max_range)
