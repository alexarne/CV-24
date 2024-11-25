import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d

from Functions import *
from gaussfft import gaussfft
from fftwave import fftwave


# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.

EXERCISE = 1.3
# EXERCISE = 1.4
# EXERCISE = 1.5
# EXERCISE = 1.6
# EXERCISE = 1.7
# EXERCISE = 1.8
# EXERCISE = 2.3
# EXERCISE = 3.1
# EXERCISE = 3.2

# Fhat = np.zeros((128, 128))
# for i in range(32):
# 	Fhat[i, 0] = 1
# F = ifft2(Fhat)
# Fabsmax = np.max(np.abs(F))
# showgrey(np.real(F), True, 64, -Fabsmax, Fabsmax)
# showgrey(np.imag(F), True, 64, -Fabsmax, Fabsmax)
# showgrey(np.abs(F), True, 64, -Fabsmax, Fabsmax)
# showgrey(np.angle(F), True, 64, -np.pi, np.pi)

match EXERCISE:
	case 1.3:
		# Repeat this exercise with the coordinates p and q set to (5, 9), (9, 5), (17, 9), (17, 121),
		# (5, 1) and (125, 1) respectively. What do you observe?
		for u, v in [(5, 9), (9, 5), (17, 9), (17, 121), (5, 1), (125, 1)]:
			fftwave(u, v)
		
	case 1.4:
		F = np.concatenate([np.zeros((56,128)), np.ones((16,128)), np.zeros((56,128))])
		G = F.T
		H = F + 2*G
		showgrey(F)
		showgrey(G)
		showgrey(H)
		Fhat = fft2(F)
		Ghat = fft2(G)
		Hhat = fft2(H)
		showgrey(np.log(1 + np.abs(Fhat)))
		showgrey(np.log(1 + np.abs(Ghat)))
		showgrey(np.log(1 + np.abs(Hhat)))
		showgrey(np.log(1 + np.abs(fftshift(Hhat))))
		
	case 1.5:
		F = np.concatenate([np.zeros((56,128)), np.ones((16,128)), np.zeros((56,128))])
		G = F.T
		showgrey(F * G)
		showfs(fft2(F))
		showfs(fft2(G))
		showfs(fft2(F * G))
		showfs(1/(128*128)*np.fft.fftshift(convolve2d(fft2(F), fft2(G), mode="same", boundary="wrap")))
		
	case 1.6:
		F = np.concatenate([np.zeros((60,128)), np.ones((8,128)), np.zeros((60,128))]) * \
			np.concatenate([np.zeros((128,48)), np.ones((128,32)), np.zeros((128,48))], axis=1)
		showgrey(F)
		showfs(fft2(F))
		
	case 1.7:
		F = np.concatenate([np.zeros((60,128)), np.ones((8,128)), np.zeros((60,128))]) * \
			np.concatenate([np.zeros((128,48)), np.ones((128,32)), np.zeros((128,48))], axis=1)
		fig = plt.figure(1)
		for i, alpha in enumerate([0, 30, 45, 60, 90]):
			G = rot(F, alpha)
			fig.add_subplot(5, 3, i*3+1).set_title("alpha = " + str(alpha))
			showgrey(G, display=False)

			Ghat = fft2(G)
			fig.add_subplot(5, 3, i*3+2).set_title("alpha = " + str(alpha))
			showfs(Ghat, display=False)

			Hhat = fftshift(rot(Ghat), -alpha)
			fig.add_subplot(5, 3, i*3+3).set_title("alpha = " + str(alpha))
			showgrey(np.log(1 + abs(Hhat)), display=False)
		plt.show()
		
	case 1.8:
		images = [
			np.load("Images-npy/phonecalc128.npy"),
			np.load("Images-npy/few128.npy"),
			np.load("Images-npy/nallo128.npy")
		]
		fig = plt.figure(1)
		for i, img in enumerate(images):
			fig.add_subplot(3, 3, i*3+1).set_title("Raw")
			showgrey(img, display=False)

			fig.add_subplot(3, 3, i*3+2).set_title("Pow")
			data = pow2image(img, a=1e-3)
			showgrey(data, display=False)
			
			fig.add_subplot(3, 3, i*3+3).set_title("Randphase")
			data = randphaseimage(img)
			showgrey(data, display=False)
		plt.show()
		
	case 2.3:
		T = [0.1, 0.3, 1.0, 10.0, 100.0]
		for t in T:
			pic = deltafcn(128, 128)
			gauss_fft = gaussfft(pic, t)
			discgauss_fft = discgaussfft(pic, t)
			print(f"t = {t}:")
			print(variance(gauss_fft))
			print(f"disc t = {t}:")
			print(variance(discgauss_fft))

		T = [1.0, 4.0, 16.0, 64.0, 256.0]
		images = [
			np.load("Images-npy/phonecalc128.npy"),
			np.load("Images-npy/few128.npy"),
			np.load("Images-npy/nallo128.npy")
		]
		fig = plt.figure(1)
		for i, img in enumerate(images):
			fig.add_subplot(3, 6, i*6+1).set_title("Raw")
			showgrey(img, display=False)
			for j, t in enumerate(T):
				fig.add_subplot(3, 6, i*6+2+j).set_title("t = " + str(t))
				showgrey(gaussfft(img, t), display=False)
		plt.show()
		
	case 3.1:
		gauss_param = [0.5, 1.0, 1.5, 2.0]
		med_param = [2, 4, 6, 8]
		lowpass_param = [0.1, 0.125, 0.15, 0.175]
		collection = [gauss_param, med_param, lowpass_param]

		def plotComparison(img, title):
			fig = plt.figure(1)

			fig.add_subplot(4, 4, 1).set_title("Raw")
			office = np.load("Images-npy/office256.npy")
			showgrey(office, display=False)

			fig.add_subplot(4, 4, 2).set_title(title)
			showgrey(img, display=False)

			for i, params in enumerate(collection):
				for j, t in enumerate(params):
					match i:
						case 0:
							s = f"Gauss t={t}"
							img2 = discgaussfft(img, t)
						case 1:
							s = f"Median w={t}"
							img2 = medfilt(img, t)
						case 2:
							s = f"Lowpass cutoff={t}"
							img2 = ideal(img, t)
					fig.add_subplot(4, 4, 5 + i*4 + j).set_title(s)
					showgrey(img2, display=False)
			plt.show()

		office = np.load("Images-npy/office256.npy")
		add = gaussnoise(office, 16)
		sap = sapnoise(office, 0.1, 255)
		plotComparison(add, "Gauss Noise")
		plotComparison(sap, "Sap Noise")
		
	case 3.2:
		img = np.load("Images-npy/phonecalc256.npy")
		smoothimg = img
		N = 5
		f = plt.figure()
		f.subplots_adjust(wspace=0, hspace=0)
		for i in range(N):
			if i>0: # generate subsampled versions
				img = rawsubsample(img)
				# smoothimg = gaussfft(smoothimg, 0.5) # <call_your_filter_here>(smoothimg, <params>)
				# smoothimg = discgaussfft(smoothimg, 0.5) # <call_your_filter_here>(smoothimg, <params>)
				smoothimg = ideal(smoothimg, 0.25) # <call_your_filter_here>(smoothimg, <params>)
				smoothimg = rawsubsample(smoothimg)
			f.add_subplot(2, N, i + 1)
			showgrey(img, False)
			f.add_subplot(2, N, i + N + 1)
			showgrey(smoothimg, False)
		plt.show()

# Exercise 1.3
if 0:
	print("That was a stupid idea")