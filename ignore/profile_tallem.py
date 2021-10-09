from tallem import tallem_transform
from tallem.datasets import mobius_band
import line_profiler

M = mobius_band(embed=6)
X = M['points']
f = M['parameters'][:,0]

profile = line_profiler.LineProfiler()
profile.add_function(tallem_transform)
profile.enable_by_count()
Y = tallem_transform(X, f, D=3)
profile.print_stats()


# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(Y[:,0], Y[:,1], Y[:,2], marker='o', c=f)