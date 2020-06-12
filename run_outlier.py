from library.stats.distributions import *
from library.plotter import Plotter
from library.vector import Vector

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

eps = 0.0000005
sample_size = 500
inc_spread = 100
# pdf_set = [(0.9, 0, 5), (0.05, -50, 3), (0.05, 35, 1)]

data, pdf_set = d.generate_data_with_outliers(
  mean=[100, 400],
  outlier_amount=[0.05, 0.20],
  outlier_left_skew=[0.99],
  N = [sample_size],
  inc = inc_spread)
# real_points = [Vector([x,0]) for x in data]
# plot_points = [Vector([p[0], p[1]]) for p in points if p[1] > eps]

# y_sum = 0
# for y in points:
#   y_sum += y[1]
# y_sum/=inc_spread
# print("sum = ", y_sum)
# xs  = list(range(len(data)))
# random.shuffle(xs)

# data = [(d[0], d[1]) for d in zip(xs, data)]
data.sort()
data = [(d, 0) for d in data]
buffer = 0.2*max(abs(data[0][0]), abs(data[-1][0]))
distribution_curve = d.get_combined_weighted_pdf_plot_points(data[0][0]-buffer, data[-1][0]+buffer, 1/inc_spread, pdf_set)

y_sum = 0
for point in distribution_curve:
  y_sum += point[1]
y_sum/=inc_spread
print("sum = ", y_sum)

p.plot_data_sets([data, distribution_curve], show=True)
p.save("dist_outliers")


# pdf_set = [(0.8, 0.5), (0.2, 1.5)]
# pdf_set = [(0.8, 0, 2), (0.1, -30, -25), (0.1, 15, 20)]
# pdf_set = [(0.5,3,0.5), (0.5, 2, 0.5)]

# real_points = [Vector([x,0]) for x in d.generate_weighted_pdfs(pdf_set, sample_size)]

# distribution points
# points = d.get_combined_weighted_pdf_plot_points(-75, 50, 1/inc_spread, pdf_set)
# points = [Vector([p[0], p[1]]) for p in points if p[1] > eps]

# p.plot_data_sets([points, real_points], show=True)
# p.save("dist_outliers")
