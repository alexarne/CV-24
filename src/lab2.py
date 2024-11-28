import numpy as np
#from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, convolve2d
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
	dx = np.array([
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 1/2, 0, -1/2, 0],
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
	])
	dxx = np.array([
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 1, -2, 1, 0],
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
	])
	dy = np.array([
		[0, 0, 0, 0, 0],
		[0, 0, 1/2, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 0, -1/2, 0, 0],
		[0, 0, 0, 0, 0],
	])
	dyy = np.array([
		[0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, -2, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0],
	])
	dxy = convolve2d(dx, dy, shape)

	Lx = convolve2d(inpic, dx, shape)
	Ly = convolve2d(inpic, dy, shape)
	Lxx = convolve2d(inpic, dxx, shape)
	Lyy = convolve2d(inpic, dyy, shape)
	Lxy = convolve2d(inpic, dxy, shape)
	Lvv = Lx**2*Lxx + 2*Lx*Ly*Lxy + Ly**2*Lyy
	return Lvv

def Lvvvtilde(inpic, shape = 'same'):
	# ...
	dx = np.array([
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 1/2, 0, -1/2, 0],
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
	])
	dxx = np.array([
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 1, -2, 1, 0],
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
	])
	dy = np.array([
		[0, 0, 0, 0, 0],
		[0, 0, 1/2, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 0, -1/2, 0, 0],
		[0, 0, 0, 0, 0],
	])
	dyy = np.array([
		[0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, -2, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0],
	])
	dxxx = convolve2d(dx, dxx, shape)
	dyyy = convolve2d(dy, dyy, shape)
	dxxy = convolve2d(dxx, dy, shape)
	dxyy = convolve2d(dx, dyy, shape)

	# [x, y] = np.meshgrid(range(-5, 6), range(-5, 6))
	# print(convolve2d(x**3, dxxx, "valid"))
	# print(convolve2d(x**3, dxx, "valid"))
	# print(convolve2d(x**3, dx, "valid"))
	# print(convolve2d(x**2*y, dxxy, "valid"))
	# print(convolve2d(y**3, dxxx, "valid"))

	Lx = convolve2d(inpic, dx, shape)
	Ly = convolve2d(inpic, dy, shape)
	Lxxx = convolve2d(inpic, dxxx, shape)
	Lyyy = convolve2d(inpic, dyyy, shape)
	Lxxy = convolve2d(inpic, dxxy, shape)
	Lxyy = convolve2d(inpic, dxyy, shape)
	Lvvv = Lx**3*Lxxx + 3*Lx**2*Ly*Lxxy + 3*Lx*Ly**2*Lxyy + Ly**3*Lyyy
	return Lvvv

def extractedge(inpic, scale, threshold=0, shape="same"):
	# ...
	img = discgaussfft(inpic, scale)
	img_Lv = Lv(img)
	img_Lvv = Lvvtilde(img)
	img_Lvvv = Lvvvtilde(img)
	mask = img_Lvvv < 0
	curves = zerocrosscurves(img_Lvv, mask)
	mask = img_Lv > threshold
	curves = thresholdcurves(curves, mask)
	return curves
        
def houghline(curves, magnitude, nrho, ntheta, threshold, nlines = 20, verbose = False):
	# ...
	acc = np.zeros((nrho, ntheta))
	theta_values = np.linspace(-np.pi/2, np.pi/2, ntheta)
	sz_y, sz_x = magnitude.shape
	max_rho = np.sqrt(sz_y**2 + sz_x**2)
	rho_values = np.linspace(-max_rho, max_rho, nrho)

	for y, x in zip(curves[0], curves[1]):
		if magnitude[y, x] < threshold:
			continue
		# vote
		for index_theta, theta in enumerate(theta_values):
			rho = x*np.cos(theta) + y*np.sin(theta)
			index_rho = (np.abs(rho_values - rho)).argmin()
			acc[index_rho, index_theta] += 1

	# acc = discgaussfft(acc, 5)
	pos, value, _ = locmax8(acc)
	indexvector = np.argsort(value)[-nlines:]
	pos = pos[indexvector]
	linepar = []
	for thetaidxacc, rhoidxacc in pos:
		theta = theta_values[thetaidxacc]
		rho = rho_values[rhoidxacc]
		linepar.append([rho, theta])
		if verbose >= 2:
			print(f"Found line\n\trho={rho}\n\ttheta={theta}")

	return linepar, acc

def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines = 20, verbose = False):
	# ...
	curves = extractedge(pic, scale, gradmagnthreshold)
	magnitude = Lv(pic)
	linepar, acc = houghline(curves, magnitude, nrho, ntheta, gradmagnthreshold, nlines, verbose)
	sz_y, sz_x = magnitude.shape

	if verbose >= 1:
		fig = plt.figure(1)
		fig.add_subplot(1, 3, 1).set_title("Edgecurves")
		overlaycurves(pic, curves)

		fig.add_subplot(1, 3, 2).set_title("Hough space")
		showgrey(acc, display=False)

		fig.add_subplot(1, 3, 3).set_title("Image space")
		showgrey(pic, display=False)
		for rho, theta in linepar:
			x0 = 0
			y0 = rho / np.sin(theta)
			dx = sz_x
			dy = (rho - dx*np.cos(theta)) / np.sin(theta) - y0
			if verbose >= 2:
				print(f"from x0={x0} y0={y0} to x1={x0+dx} y1={y0+dy}")
			plt.plot([x0-dx, x0, x0+dx], [y0-dy, y0, y0+dy], "r-")

		plt.show()

	
	return linepar, acc
		


# EXERCISE = 1
# EXERCISE = 2
# EXERCISE = 4
# EXERCISE = 5
EXERCISE = 6

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

		plotThresholdHistogram(tools, gradmagntools, "gradmagntools")
		plotThresholdHistogram(tools_blurred, gradmagntools_blurred, "gradmagntools_blurred")
		plotThresholdHistogram(house, gradmagnhouse, "gradmagnhouse")
		plotThresholdHistogram(house_blurred, gradmagnhouse_blurred, "gradmagnhouse_blurred")

	case 4:
		scales = [0.0001, 1, 4, 16, 64]

		house = np.load("Images-npy/godthem256.npy")
		fig = plt.figure(1)
		fig.add_subplot(2, 3, 1).set_title("Raw")
		showgrey(house, display=False)
		for i, scale in enumerate(scales):
			fig.add_subplot(2, 3, 2+i).set_title(f"scale={scale}")
			showgrey(contour(Lvvtilde(discgaussfft(house, scale), "same")), display=False)
		plt.show()

		tools = np.load("Images-npy/few256.npy")
		fig = plt.figure(1)
		fig.add_subplot(2, 3, 1).set_title("Raw")
		showgrey(tools, display=False)
		for i, scale in enumerate(scales):
			fig.add_subplot(2, 3, 2+i).set_title(f"scale={scale}")
			showgrey((Lvvvtilde(discgaussfft(tools, scale), "same")<0).astype(int), display=False)
		plt.show()

	case 5:
		scales = [0.0001, 1, 4, 16, 64]
		house = np.load("Images-npy/godthem256.npy")
		tools = np.load("Images-npy/few256.npy")
		threshold = 10
		
		# fig = plt.figure(1)
		# for i, scale in enumerate(scales):
		# 	edgecurves = extractedge(house, scale, threshold)
		# 	fig.add_subplot(2, 5, 1+i).set_title(f"scale={scale}")
		# 	overlaycurves(house, edgecurves)
		# 	edgecurves = extractedge(tools, scale, threshold)
		# 	fig.add_subplot(2, 5, 6+i).set_title(f"scale={scale}")
		# 	overlaycurves(tools, edgecurves)
		# plt.show()

		fig = plt.figure(1)
		scale = 5
		threshold = 35
		fig.add_subplot(1, 2, 1).set_title(f"scale={scale}, threshold={threshold}")
		edgecurves = extractedge(house, scale, threshold)
		overlaycurves(house, edgecurves)
		scale = 8
		threshold = 35
		fig.add_subplot(1, 2, 2).set_title(f"scale={scale}, threshold={threshold}")
		edgecurves = extractedge(tools, scale, threshold)
		overlaycurves(tools, edgecurves)
		plt.show()

	case 6:
		testimage1 = np.load("Images-npy/triangle128.npy")
		smalltest1 = binsubsample(testimage1)
		testimage2 = np.load("Images-npy/houghtest256.npy")
		smalltest2 = binsubsample(binsubsample(testimage2))
		houghedgeline(smalltest1, 1, 5, 100, 100, 5, 2)
		houghedgeline(testimage1, 1, 5, 100, 100, 5, 2)
		houghedgeline(smalltest2, 1, 20, 100, 100, 20, 2)
		houghedgeline(testimage2, 1, 20, 100, 100, 20, 2)

