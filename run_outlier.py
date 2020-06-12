from library.stats.distributions import *
from library.stats.outlier_detection import OutlierDetection
from library.plotter import Plotter


# e = ExponentialDistribution()
# g = GammaDistribution()
# u = UniformDistribution()
# n = NormalDistribution()
# d = Distribution()

p = Plotter()


# dist = {}
# generated = n.generate_with_outliers(0, 1, 50000)

# generated = d.generate_random(100000)['data'].tolist()

# for g in generated:
#   rounded = g
#   rounded = round(g, 1)
#   print(g, rounded)
#   if rounded not in dist:
#     dist[rounded] = 0
#   dist[rounded] += 1

# print(dist)

# points = []
# for d in dist:
#   points.append(Vector([d, dist[d]]))

# points = [Vector([x,y]) for x, y in zip(, ys)]

# print(distribution)

# low_outlier_dist = NormalDistribution(-5, 2)


# data_points = data_dist.get_plot_points(-15, 15, 1/inc_spread)
# low_outlier_points = low_outlier_dist.get_plot_points(-15, 15, 1/inc_spread)

# data_weight = 0.9
# low_outlier_weight = 0.1

# points = []
# for i in range(len(data_points)):
#   new_x = data_points[i][0]
#   new_y = data_weight * data_points[i][1] + low_outlier_weight * low_outlier_points[i][1]
#   new_point = (new_x, new_y)
#   points.append(new_point)



'''
n = NormalDistribution()
inc_spread = 10
pdf_set = [(0, 1, 0.9), (-50, 3, 0.05), (35, 1, 0.05)]
points = n.get_combined_weighted_pdf_plot_points(-75, 50, 1/inc_spread, pdf_set)


points = [Vector([p[0], p[1]]) for p in points]

p.plot(points, show=True)

y_sum = 0
for p in points:
  y_sum += p.y()
y_sum/=inc_spread
print("sum = ", y_sum)
'''

n = NormalDistribution()
e = ExponentialDistribution()
u = UniformDistribution()
g = GammaDistribution()

d = n

sample_size = 2500
granurality = 100
# pdf_set = [(0.9, 0, 5), (0.05, -50, 3), (0.05, 35, 1)]

data, pdf_set = d.generate_data_with_outliers(
  mean=[120, 500],
  outlier_amount=[0.20],
  num_outlier_sources = [15,20],
  N = [sample_size])

o = OutlierDetection()
cleaned_data = o.remove_outliers(np.array(data), "MAD")

# data = [(d, 0) for d in data]

cleaned_set = set(cleaned_data)

removed_points = [(d, 0) for d in data if d not in cleaned_set]

cleaned_data = [(d,0) for d in cleaned_data]

distribution_curve = d.get_combined_weighted_pdf_plot_points(1/granurality, pdf_set)

y_sum = 0
for point in distribution_curve:
  y_sum += point[1]
y_sum/=granurality
print("sum = ", y_sum)

p.plot_data_sets([cleaned_data, distribution_curve, removed_points], show=True)
p.save("dist_outliers")
