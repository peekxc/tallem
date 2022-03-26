import numpy as np
from tallem.datasets import white_bars, plot_images


bar, c = white_bars(n_pixels=17, r=0.20, sigma=0.50)
samples = []
for d in np.linspace(-0.5, 0.5, num=38, endpoint=True):
  for theta in np.linspace(0, np.pi, num=11, endpoint=True):
    samples.append(np.ravel(bar(theta, d)).flatten())
samples = np.vstack(samples)
fig, ax = plot_images(samples, shape=(17,17), max_val=c, layout=(38,11))



