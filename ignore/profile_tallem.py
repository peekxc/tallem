from tallem import tallem_transform
from tallem.datasets import mobius_band
import line_profiler

M = mobius_band(embed=6)
X = M['points']
f = M['parameters'][:,0]

profile = line_profiler.LineProfiler()
profile.add_function(tallem_transform)
profile.enable_by_count()
tallem_transform(X, f, D=3)
profile.print_stats()
