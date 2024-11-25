import numpy as np
#from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft

# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.
        
def deltax():
	# ....
	dxmask = np.array([
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1],
	])
	return dxmask

def deltay():
	# ....
	dymask = np.array([
		[1, 2, 1],
		[0, 0, 0],
		[-1, -2, -1],
	])
	return dymask

def Lv(inpic, shape = 'same'):
	# ...
	dxmask = deltax()
	dymask = deltay()
	Lx = convolve2d(inpic, dxmask, shape)
	Ly = convolve2d(inpic, dymask, shape)
	result = np.sqrt(Lx**2 + Ly**2)
	return result

def Lvvtilde(inpic, shape = 'same'):
	# ...
	return result

def Lvvvtilde(inpic, shape = 'same'):
	# ...
	return result

def extractedge(inpic, scale, threshold, shape):
	# ...
	return contours
        
def houghline(curves, magnitude, nrho, ntheta, threshold, nlines = 20, verbose = False):
	# ...
	return linepar, acc

def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines = 20, verbose = False):
	# ...
	return linepar, acc
		

def plotThresholdHistogram(source, grad, title):
	fig = plt.figure(1)
	fig.add_subplot(2, 4, 2).set_title("Raw")
	showgrey(source, display=False)
	fig.add_subplot(2, 4, 3).set_title(title)
	showgrey(grad, display=False)

	thresholds = [50, 100, 150, 200]
	for i, threshold in enumerate(thresholds):
		fig.add_subplot(2, 4, 5+i).set_title(f"Threshold = {threshold}")
		showgrey((grad > threshold).astype(int), display=False)
		
	plt.show()

	hist, bins = np.histogram(grad, bins=70)
	plt.hist(hist, bins)
	plt.title(f"Histogram {title}")
	plt.autoscale()
	plt.show()



# EXERCISE = 1
EXERCISE = 2

match EXERCISE:
	case 1:
		tools = np.load("Images-npy/few256.npy")
		dxtools = convolve2d(tools, deltax(), "valid")
		dytools = convolve2d(tools, deltay(), "valid")

		fig = plt.figure(1)
		fig.add_subplot(1, 3, 1).set_title("Raw")
		showgrey(tools, display=False)
		fig.add_subplot(1, 3, 2).set_title("dxtools")
		showgrey(dxtools, display=False)
		fig.add_subplot(1, 3, 3).set_title("dytools")
		showgrey(dytools, display=False)
		plt.show()

		print(f"Original image size: {tools.shape}")
		print(f"dxtools image size: {dxtools.shape}")
		print(f"dytools image size: {dytools.shape}")

	case 2:
		tools = np.load("Images-npy/few256.npy")
		house = np.load("Images-npy/godthem256.npy")
		tools_blurred = discgaussfft(tools, 1)
		house_blurred = discgaussfft(house, 1)
		gradmagntools = Lv(tools, shape="valid")
		gradmagntools_blurred = Lv(tools_blurred, shape="valid")
		gradmagnhouse = Lv(house, shape="valid")
		gradmagnhouse_blurred = Lv(house_blurred, shape="valid")
		plotThresholdHistogram(tools, gradmagntools, "gradmagntools")
		plotThresholdHistogram(tools_blurred, gradmagntools_blurred, "gradmagntools_blurred")
		plotThresholdHistogram(house, gradmagnhouse, "gradmagnhouse")
		plotThresholdHistogram(house_blurred, gradmagnhouse_blurred, "gradmagnhouse_blurred")