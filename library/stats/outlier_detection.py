import math
import numpy as np
import random
import statistics
import time
import os

class OutlierDetection:

  def __init__(self):
    pass

  # using a slightly modified script from https://stackoverflow.com/questions/55351782/how-should-i-generate-outliers-randomly
  def generate(self, median=630, err=12, outlier_err=100, size_total=50, num_outliers=10):
    errs = err * np.random.rand(size_total-num_outliers) * np.random.choice((-1, 1), size_total-num_outliers)
    data = median + errs

    num_lower_outliers = int(random.random() * num_outliers)
    num_higher_outliers = num_outliers - num_lower_outliers

    lower_errs = outlier_err * np.random.rand(num_lower_outliers)
    lower_outliers = median - err - lower_errs

    upper_errs = outlier_err * np.random.rand(num_higher_outliers)
    upper_outliers = median + err + upper_errs

    data = np.concatenate((data, lower_outliers, upper_outliers))
    np.random.shuffle(data)

    return data

  def generate_random(self, size = 1000):
    # size_total = random.randint(500, 5000)
    size_total = size
    num_outliers = random.randint(1, int(0.05*size_total))

    median = random.randint(500,1000)
    err = (random.randint(5,500)/10000)*median # 5% error
    outlier_err = (random.randint(2500, 5000)/10000)*median

    num_lower_outliers = int(random.random() * num_outliers)
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


  def remove_outliers(self, values_np, method):
    values = values_np.tolist()
    sorted_values = sorted(values)
    # sorted_values = np.sort(values_np)
    # sorted_values = sorted_values.tolist()

    if method == "MAD":
      # MAD outlier detection method
      # https://en.wikipedia.org/wiki/Median_absolute_deviation

      median = self.median_sorted(sorted_values)
      sorted_values = self.remove_outliers_mad_method(sorted_values, median)

    elif method == "MEAN":
      # Assumes high leveraging points make up between 0 and 5 percent of the data
      mean_value = sum(sorted_values)/len(sorted_values)
      std_dev = self.get_std_deviation(sorted_values, mean_value)
      sorted_values = self.remove_outliers_deviation_method(sorted_values, mean_value, std_dev)

    elif method == "MEAN_MODIFIED":
      # Assumes high leveraging points make up between 0 and 5 percent of the data
      sorted_values_removed_ends = self.remove_outliers_percentile_method(sorted_values, 0.05)
      mean_value = sum(sorted_values_removed_ends)/len(sorted_values_removed_ends)
      std_dev = self.get_std_deviation(sorted_values_removed_ends, mean_value)
      sorted_values = self.remove_outliers_deviation_method(sorted_values, mean_value, std_dev)

    elif method == "MEDIAN":
      # Same as MEAN method using
      median = self.median_sorted(sorted_values)
      std_dev = self.get_std_deviation(sorted_values, median)
      sorted_values = self.remove_outliers_deviation_method(sorted_values, median, std_dev)

    elif method == "MEDIAN_MODIFIED":
      # Same as MEAN method using
      sorted_values_removed_ends = self.remove_outliers_percentile_method(sorted_values, 0.05)
      median = self.median_sorted(sorted_values_removed_ends)
      std_dev = self.get_std_deviation(sorted_values_removed_ends, median)
      sorted_values = self.remove_outliers_deviation_method(sorted_values, median, std_dev)

    elif method == "PERCENTILE_5":
      # Remove 5% off both ends of dataset
      sorted_values = self.remove_outliers_percentile_method(sorted_values, 0.05)

    elif method == "PERCENTILE_2.5":
      # Remove 2.5% off both ends of dataset
      sorted_values = self.remove_outliers_percentile_method(sorted_values, 0.025)
    
    elif method == "NEW":
      sorted_values = self.remove_outliers_new_method(sorted_values, 0.02)

    elif method == "NEW_MAD":
      sorted_values = self.remove_outliers_new_mad_method(sorted_values, 0.02, 0.07)
    return sorted_values

  def remove_outliers_percentile_method(self, values, interpercentile):
    remove_from_ends = int(interpercentile * len(values))
    return values[remove_from_ends : -remove_from_ends]

  def get_std_deviation(self, sorted_values, mean_value):
    sq_sum = 0
    for val in sorted_values: sq_sum += ((val-mean_value)**2)
    variance = sq_sum/len(sorted_values)
    return math.sqrt(variance)

  def remove_outliers_deviation_method(self, values, mean, sd):
    MAX_DEVIATION = 3.0
    remove_left = 0
    remove_right = 0

    while remove_left < len(values):
      difference_from_mean = values[remove_left] - mean
      deviation_from_mean = abs(difference_from_mean / sd)
      if deviation_from_mean <= MAX_DEVIATION: break
      remove_left += 1

    while remove_right < len(values):
      difference_from_mean = values[-remove_right - 1] - mean
      deviation_from_mean = abs(difference_from_mean / sd)
      if deviation_from_mean <= MAX_DEVIATION: break
      remove_right += 1

    if (remove_left + remove_right > len(values)): return
    return values[remove_left : len(values) - remove_right]

  # https://en.wikipedia.org/wiki/Median_absolute_deviation
  def remove_outliers_mad_method(self, values, median):
    CUT_OFF = 3
    MAD = []
    for v in values:
      MAD.append(abs(v - median))
    MAD.sort()
    MAD = self.median_sorted(MAD)

    new_list = []
    for v in values:
      modified_z_score = 0.6745 * abs(v - median) / MAD
      if modified_z_score < CUT_OFF:
        new_list.append(v)
    return new_list

  def median_sorted(self, values, left=None, right=None):
    to_return = 0
    if not left: left, right = 0, len(values)
    if right - left == 0: to_return = 0
    elif (right - left) % 2 == 1: to_return = values[int((right - left) / 2)]
    else: to_return = (values[int((right-left) / 2) - 1] + values[int((right-left) / 2)]) / 2
    return to_return

  def remove_outliers_new_method(self, values, o_star=None):
    '''
    assume values are sorted
    '''

    median = self.median_sorted(values)
    if not o_star:
      values_after_mad = self.remove_outliers_mad_method(values, median)
      o_star = abs((len(values_after_mad) - len(values))/ len(values))
    # print("afterMAD:", len(values_after_mad), "o* = ", o_star)


    eps_star = 0.0000000000000005
    # we can avoid this step by making first choice eps_k_+ and let r_initial = o_star and l_initial = 0
    
    l_prev = 0
    r_prev = o_star
    eps_k_abs = o_star / 2 # absolute value of eps_k

    while eps_k_abs > eps_star:
      choice = self.new_method_make_choice(values, median, l_prev, r_prev, eps_k_abs)  # l + eps_k
      if choice == 0:
        eps_k_abs /= 2
        continue
      eps_k = eps_k_abs if choice == 1 else -eps_k_abs
      if l_prev + eps_k >= 0 and r_prev - eps_k >= 0:
        l_prev += eps_k
        r_prev -= eps_k
      eps_k_abs /= 2

    print("Detected skewness: [{},{}]".format(l_prev / o_star * 100, r_prev / o_star * 100))
    print("lprev={}, rprev={}".format(l_prev, r_prev))
    sorted_values_removed_ends = self.remove_outliers_each_side(values, l_prev, r_prev)
    return sorted_values_removed_ends

    # median = self.median_sorted(sorted_values_removed_ends)
    # # mean = sum(sorted_values_removed_ends)/len(sorted_values_removed_ends)
    # std_dev = self.get_std_deviation(sorted_values_removed_ends, median)
    # sorted_values = self.remove_outliers_deviation_method(values, median, std_dev)
    # return sorted_values

  def remove_outliers_new_mad_method(self, values, o_star_lo, o_star_hi):
    '''
    assume values are sorted
    '''

    median = self.median_sorted(values)
    # if not o_star:
    #   values_after_mad = self.remove_outliers_mad_method(values, median)
    #   o_star = abs((len(values_after_mad) - len(values))/ len(values))
    # print("afterMAD:", len(values_after_mad), "o* = ", o_star)


    eps_star = 0.0000000000000005
    # we can avoid this step by making first choice eps_k_+ and let r_initial = o_star and l_initial = 0
    
    l_prev_lo, l_prev_hi = 0, 0
    r_prev_lo, r_prev_hi = o_star_lo, o_star_hi

    # do LOW first
    eps_k_abs = o_star_lo / 2

    while eps_k_abs > eps_star:
      choice = self.new_method_make_choice(values, median, l_prev_lo, r_prev_lo, eps_k_abs)  # l + eps_k
      if choice == 0:
        eps_k_abs /= 2
        continue
      eps_k = eps_k_abs if choice == 1 else -eps_k_abs
      if l_prev_lo + eps_k >= 0 and r_prev_lo - eps_k >= 0:
        l_prev_lo += eps_k
        r_prev_lo -= eps_k
      eps_k_abs /= 2

    # do HIGH
    eps_k_abs = o_star_hi / 2
    
    while eps_k_abs > eps_star:
      choice = self.new_method_make_choice(values, median, l_prev_hi, r_prev_hi, eps_k_abs)  # l + eps_k
      if choice == 0:
        eps_k_abs /= 2
        continue
      eps_k = eps_k_abs if choice == 1 else -eps_k_abs
      if l_prev_hi + eps_k >= 0 and r_prev_hi - eps_k >= 0:
        l_prev_hi += eps_k
        r_prev_hi -= eps_k
      eps_k_abs /= 2
    
    # remove the low estimates first
    sorted_values_removed_ends = self.remove_outliers_each_side(values, l_prev_lo, r_prev_lo)
    return self.remove_outliers_mad_method(sorted_values_removed_ends, median)


    # print("Detected skewness: [{},{}]".format(l_prev / o_star * 100, r_prev / o_star * 100))
    # print("lprev={}, rprev={}".format(l_prev, r_prev))
    return sorted_values_removed_ends



  def remove_outliers_each_side(self, values, left, right):
    fr = max(0, int(left * len(values)))
    to = min(-1, - int(right*len(values) - 1))
    to_return = values[fr:to]
    print("from:{} to {}, len={}".format(fr, to, len(to_return)))
    return to_return

  def new_method_make_choice(self, values, median, l_k, r_k, eps_k_abs):
    '''

    '''
    NO_CHOICE_MARGIN = 0.0000000005
    # print(l_k, r_k)

    l_k *= len(values)
    r_k *= len(values)
    eps_k_abs *= len(values)

    # choice 1: l + eps_k_abs, r + eps_k_abs
    l_k_1 = values[int(l_k + eps_k_abs)]
    # r_k_1 = values[int(r_k + eps_k_abs)]
    median_1 = self.median_sorted(values, l_k_1, median)
    diff_median1 = abs(abs(median) - abs(median_1))
    # choice 2: l - eps_k_abs, r - eps_k_abs
    # l_k_2 = values[int(-l_k - eps_k_abs)]
    r_k_2 = values[int(-r_k - eps_k_abs)]
    median_2 = self.median_sorted(values, median, r_k_2)
    diff_median2 = abs(abs(median) - abs(median_2))
    
    choice = None

    # look at difference from median
    choice_1_difference = abs(median - l_k_1)
    choice_2_difference = abs(median - r_k_2)
    # print("clos=", abs(1 - diff_median1 / diff_median2))

    if diff_median1 == diff_median2 or abs(1 - diff_median1 / diff_median2) < NO_CHOICE_MARGIN: choice = 0
    elif diff_median1 < diff_median2: choice = 1
    else: choice = 2

    # if choice_1_difference == choice_2_difference or abs(1 - choice_1_difference/choice_2_difference) < NO_CHOICE_MARGIN: choice = 0
    # elif choice_1_difference > choice_2_difference: choice = 1
    # else: choice = 2


    # print(diff_median1, diff_median2, choice)

    return choice
    # print(choice_1_difference, choice_2_difference, choice_1_difference/choice_2_difference, l_k, r_k, l_k_1, r_k_2)

    # if diff_median1 == diff_median2 or abs(1 - diff_median1 / diff_median2) < NO_CHOICE_MARGIN: choice = 0
    # elif diff_median1 < diff_median2: choice = 1
    # else: choice = 2

    # print("choice=", choice)
    # return choice



    # technique 2:
    # initial sums arr[0] -> arr[l_k] and arr[-1] -> arr[-r_k - 1]
    # choice 1: sum an array from arr[0] -> arr[l_k_1] and from arr[-1] -> arr[-r_k_1 - 1]
    # choice 2: sum arr[0] -> arr[l_k_2] and from arr[-1] -> arr[-r_k_2 - 1]
    

    



  def write_to_csv(self, file_name, data):
    ext = '.csv'
    file_to_write = file_name + ext

    if os.path.exists(file_to_write):
      append_write = 'a' # append if already exists
    else:
      append_write = 'w' # make a new file if not

    new_data = ','.join(data) + '\n'
    with open(file_to_write, append_write) as myfile:
      myfile.write(new_data)


  def run_and_write_to_csv(self, trials=1000, SAMPLE_SIZE=1000):

    methods = ["NONE", "PERCENTILE_5", "PERCENTILE_2.5", "MEAN", "MEAN_MODIFIED", "MEDIAN", "MEDIAN_MODIFIED", "MAD"]

    errors = {}
    times = {}
    errors_generated_dict = {}
    errors_detected_dict = {}
    value_range = {}
    for m in methods:
      errors[m] = []
      times[m] = []
      value_range[m] = [0,0,0]
      errors_generated_dict[m] = 0
      errors_detected_dict[m] = 0

    average_median = 0
    average_range = [0,0]
    average_outlier_range = [0,0]
    total_num_values = 0

    for trial in range(trials):

      generated_data = self.generate_random(SAMPLE_SIZE)

      num_values = generated_data['num_values']
      num_outliers = generated_data['num_outliers']
      median = generated_data['median']
      data = generated_data['data']

      correct_range = generated_data['correct_range']
      average_range[0] += correct_range[0]
      average_range[1] += correct_range[1]

      outlier_range = generated_data['outlier_range']
      average_outlier_range[0] += outlier_range[0]
      average_outlier_range[1] += outlier_range[1]

      average_median += median
      total_num_values += num_values

      for method in random.sample(methods, len(methods)):
        t1 = time.time()
        data_without_outliers = self.remove_outliers(data, method)
        t2 = time.time()

        # sort data and extract range of values
        data_without_outliers.sort()
        value_range[method][0] += self.median_sorted(data_without_outliers)
        value_range[method][1] += data_without_outliers[0]
        value_range[method][2] += data_without_outliers[-1]

        errors_detected = abs((len(data) - len(data_without_outliers)))
        errors_generated = num_outliers

        percentage_error = abs(100.0 * (float(errors_detected-errors_generated)/float(errors_generated)))
        # percentage_error = abs(100 * ( (len(data_without_outliers) - (num_values - num_outliers))/((num_values - num_outliers))))
        time_executed = t2-t1

        errors[method].append(percentage_error)
        times[method].append(time_executed)
        errors_generated_dict[method] += errors_generated
        errors_detected_dict[method] += errors_detected



    # writing to csv:
    # true values:

    root_folder = 'data/err80outlier120_150_5/'
    folder = root_folder + 'performance/'
    method = 'NO OUTLIERS'
    median_avg = "{0:.3f}".format(average_median/trials)
    lower_avg = "{0:.3f}".format(average_range[0]/trials)
    upper_avg = "{0:.3f}".format(average_range[1]/trials)
    data_to_write = [str(SAMPLE_SIZE), median_avg, lower_avg, upper_avg]
    self.write_to_csv(folder + method, data_to_write)

    # outlier values:
    folder = root_folder + 'performance/'
    method = 'ALL OUTLIERS'
    median_avg = "{0:.3f}".format(average_median/trials)
    lower_avg = "{0:.3f}".format(average_outlier_range[0]/trials)
    upper_avg = "{0:.3f}".format(average_outlier_range[1]/trials)
    data_to_write = [str(SAMPLE_SIZE), median_avg, lower_avg, upper_avg]
    self.write_to_csv(folder + method, data_to_write)

    for method in methods:
      # error info
      folder = root_folder + 'performance/'
      vrange = value_range[method]
      median_avg = "{0:.3f}".format(vrange[0]/trials)
      lower_avg = "{0:.3f}".format(vrange[1]/trials)
      upper_avg = "{0:.3f}".format(vrange[2]/trials)
      data_to_write = [str(SAMPLE_SIZE), median_avg, lower_avg, upper_avg]
      self.write_to_csv(folder + method, data_to_write)

      # runtime
      folder = root_folder + 'runtime/'
      time_list = sorted(times[method])
      time_list_sum = sum(time_list)
      time_mean = time_list_sum / len(time_list)
      time_sd = o.get_std_deviation(time_list, time_mean)
      time_ci95 = 1.96 * time_sd / math.sqrt(len(time_list))

      time_list_sum, time_ci95/time_mean * time_list_sum

      median_avg = "{0:.3f}".format(time_list_sum)
      lower_avg = "{0:.3f}".format(time_list_sum - time_ci95/time_mean * time_list_sum)
      upper_avg = "{0:.3f}".format(time_list_sum + time_ci95/time_mean * time_list_sum)

      data_to_write = [str(SAMPLE_SIZE), median_avg, lower_avg, upper_avg]
      self.write_to_csv(folder + method, data_to_write)

      # accuracy
      folder = root_folder + 'accuracy/'

      percentage_error_list = sorted(errors[method])
      percentage_error_mean = sum(percentage_error_list) / len(percentage_error_list)
      percentage_error_sd = o.get_std_deviation(percentage_error_list, percentage_error_mean)
      error_ci95 = 1.96 * percentage_error_sd / math.sqrt(len(percentage_error_list))

      median_avg = "{0:.3f}".format(percentage_error_mean)
      lower_avg = "{0:.3f}".format(percentage_error_mean - error_ci95)
      upper_avg = "{0:.3f}".format(percentage_error_mean + error_ci95)

      data_to_write = [str(SAMPLE_SIZE), median_avg, lower_avg, upper_avg]
      self.write_to_csv(folder + method, data_to_write)



# o = OutlierDetection()
# trials = 1000
# for i in range(500, 10001, 500):
#   print("Size =", i)
#   o.run_and_write_to_csv(SAMPLE_SIZE=i)



# for method in methods:
#   percentage_error_list = sorted(errors[method])
#   percentage_error_mean = sum(percentage_error_list) / len(percentage_error_list)
#   percentage_error_sd = o.get_std_deviation(percentage_error_list, percentage_error_mean)
#   error_ci95 = 1.96 * percentage_error_sd / math.sqrt(len(percentage_error_list))
#   errors_generated_total = errors_generated_dict[method]
#   errors_detected_total = errors_detected_dict[method]

#   time_list = sorted(times[method])
#   time_list_sum = sum(time_list)
#   time_mean = time_list_sum / len(time_list)
#   time_sd = o.get_std_deviation(time_list, time_mean)
#   time_ci95 = 1.96 * time_sd / math.sqrt(len(time_list))

#   print("Method: {0} ({1} trials)\
#     \ntotal outliers generated: {2}\
#     \ntotal outliers detected: {3}\
#     \nerror %: {4:.3f} ± {5:.3f}\
#     \ntotal time executed (seconds) = {6:.3f} ± {7:.3f}\
#     \nrange = [{8:.3f}, {9:.3f}]\
#     \n".format(\
#     method,\
#     trials,\
#     errors_generated_total,\
#     errors_detected_total,\
#     percentage_error_mean, error_ci95 ,\
#     time_list_sum, time_ci95/time_mean * time_list_sum, \
#     value_range[method][1]/trials, value_range[method][2]/trials))

# print("True range = [{0:.3f}, {1:.3f}, {2:.3f}]".format(average_range[0]/trials, average_median/trials, average_range[1]/trials))
# print("Outlier range = [{0:.3f}, {1:.3f}, {2:.3f}]".format(average_outlier_range[0]/trials, average_median/trials, average_outlier_range[1]/trials))






# for method in methods:
#   t1 = time.time()
#   o = OutlierDetection()
#   trials = 10000

#   for _ in range(trials):

#     num_values = random.randint(500, 1000)
#     num_outliers = random.randint(1, int(0.05*num_values))

#     median = random.randint(500,1000)
#     standard_error = (random.randint(5,500)/10000)*median # 5% error
#     outlier_error = (random.randint(2500, 5000)/10000)*median

#     data = o.generate(median, standard_error, outlier_error, num_values, num_outliers)
#     data_without_outliers = o.remove_outliers(data, method)

#     # error_count = num_outliers - (len(data) - len(data_without_outliers))
#     errors_detected = abs((len(data) - len(data_without_outliers)))
#     errors_generated = num_outliers

#     percentage_error = abs(100*((errors_detected-errors_generated)/errors_generated))
#     errors[method].append(percentage_error)

#   percentage_error_list = sorted(errors[method])

#   percentage_error_mean = sum(percentage_error_list) / len(percentage_error_list)
#   percentage_error_sd = o.get_std_deviation(percentage_error_list, percentage_error_mean)

#   ci95 = 1.96 * percentage_error_sd / math.sqrt(len(percentage_error_list))

#   t2 = time.time()
#   print("Method:{}\noutliers in {} trials:\ngenerated: {}\ndetected: {}\nerr%: {}±{}\ntime executed = {}\n".format(\
#     method,\
#     trials,\
#     errors_generated,\
#     errors_detected,\
#     percentage_error_mean, ci95 ,\
#     t2-t1))
