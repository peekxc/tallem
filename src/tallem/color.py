
import numpy as np
from numpy.typing import ArrayLike
from typing import *
from .color_constants import COLORS
# Largely from: https://bsouthga.dev/posts/color-gradients-with-python

def colors_to_hex(x):
	''' Given an iterable of color names, rgb values, or a hex strings, returns a list of the corresponding hexadecimal color representation '''
	def convert_color(c):
		if isinstance(c, str):
			return(c if c[0] == "#" else rgb_to_hex(COLORS[c]))
		elif len(c) == 3:
			return(rgb_to_hex(c))
		else: 
			raise ValueError("Invalid input detected")
	return(np.array([convert_color(c) for c in x]))


def hex_to_rgb(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def rgb_to_hex(rgb):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  rgb = [int(x) for x in rgb]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in rgb])


def color_dict(gradient):
  ''' Converts a list of RGB lists to a dictionary of colors in RGB and hex form. '''
  return {"hex":[rgb_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def _linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
	''' returns a gradient list of (n) colors between
		two hex colors. start_hex and finish_hex
		should be the full six-digit color string,
		inlcuding the number sign ("#FFFFFF") '''
	# Starting and ending colors in RGB form
	s = hex_to_rgb(start_hex)
	f = hex_to_rgb(finish_hex)
	# Initilize a list of the output colors with the starting color
	RGB_list = [s]
	# Calculate a color at each evenly spaced value of t from 1 to n
	for t in range(1, n):
		# Interpolate RGB vector for color at the current value of t
		curr_vector = [
			int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
			for j in range(3)
		]
		# Add it to our list of output colors
		RGB_list.append(curr_vector)
	return color_dict(RGB_list)


def linear_gradient(colors, n):
	''' returns a list of colors forming linear gradients between
			all sequential pairs of colors. "n" specifies the total
			number of desired output colors '''
	colors = colors_to_hex(colors)
	# The number of colors per individual linear gradient
	n_out = int(float(n) / (len(colors) - 1))
	# returns dictionary defined by color_dict()
	gradient_dict = _linear_gradient(colors[0], colors[1], n_out)

	if len(colors) > 1:
		for col in range(1, len(colors) - 1):
			next = _linear_gradient(colors[col], colors[col+1], n_out)
			for k in ("hex", "r", "g", "b"):
				gradient_dict[k] += next[k][1:] # Exclude first point to avoid duplicates
	return gradient_dict

def hist_equalize(x, number_bins=100000):
	h, bins = np.histogram(x.flatten(), number_bins, density=True)
	cdf = h.cumsum() # cumulative distribution function
	cdf = np.max(x) * cdf / cdf[-1] # normalize
	return(np.interp(x.flatten(), bins[:-1], cdf))

def bin_color(x: ArrayLike, color_pal: List, min_x = "default", max_x = "default", scaling="linear", **kwargs):
	''' Bins non-negative values 'x' into appropriately scaled bins matched with the given color range. '''
	x_min = np.min(x) if min_x == "default" else min_x
	x_max = np.max(x) if max_x == "default" else max_x
	assert isinstance(x_min, float) and isinstance(x_max, float)
	if scaling == "linear":
		x = x
	elif scaling == "logarithmic":
		x = np.log(x + x_min + 1.0)
		x_min = np.log(x_min + 1.0)
		x_max = np.log(x_max + 1.0)
	elif scaling == "equalize":
		x = hist_equalize(x, **kwargs)
		x_min, x_max = np.min(x), np.max(x)
	else:
		raise ValueError(f"Unknown scaling option '{scaling}' passed. Must be one of 'linear', 'logarithmic', or 'equalize'. ")
	ind = np.digitize(x, bins=np.linspace(x_min, x_max, len(color_pal)))
	ind = np.minimum(ind, len(color_pal)-1) ## bound from above
	ind = np.maximum(ind, 0)								## bound from below
	return(np.asanyarray(color_pal)[ind])