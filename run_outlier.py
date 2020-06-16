from library.stats.distributions import *
from library.stats.outlier_detection import OutlierDetection
from library.plotter import Plotter
import time


p = Plotter()
n = NormalDistribution()
e = ExponentialDistribution()
u = UniformDistribution()
g = GammaDistribution()
o = OutlierDetection()
d = n

MEAN = [200, 600]
SAMPLE_SIZE = 1000000
OUTLIER_FRAC = [0.05]
FIRST_SKEW = [0.7]
NUM_OUTLIER_SOURCES = [2]
OUTLIER_SIDES = [1, -1]
granularity = 100

# pdf_set = [(0.9, 0, 5), (0.05, -50, 3), (0.05, 35, 1)]

data, pdf_set = d.generate_data_with_outliers(
  mean=MEAN,
  outlier_amount=OUTLIER_FRAC,
  outlier_first_skew=FIRST_SKEW,
  num_outlier_sources = NUM_OUTLIER_SOURCES,
  outlier_sides = OUTLIER_SIDES,
  N = [SAMPLE_SIZE])
  
t1 = time.time()
cleaned_data = o.remove_outliers(np.array(data), "NEW_MAD")
t2 = time.time()


cleaned_set = set(cleaned_data)

removed_points = [(d, 0) for d in data if d not in cleaned_set]

cleaned_data = [(d,0) for d in cleaned_data]

distribution_curve = d.get_combined_weighted_pdf_plot_points(1/granularity, pdf_set)

y_sum = 0
for point in distribution_curve:
  y_sum += point[1]
y_sum/=granularity
print("sum = ", y_sum)

percentage_removed = len(removed_points)/SAMPLE_SIZE
percentage_error = abs(percentage_removed - OUTLIER_FRAC[0]) / OUTLIER_FRAC[0]
print("perr = {}%".format(100*round(percentage_error,4)))
print("detected: {} compared to {}".format(len(removed_points), OUTLIER_FRAC[0]*SAMPLE_SIZE))
print("time = {}".format(t2-t1))

p.plot_data_sets([cleaned_data, distribution_curve, removed_points], show=True)
# p.save("dist_outliers")
