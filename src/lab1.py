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

EXERCISE = 1.9

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

			Hhat = rot(fftshift(Ghat), -alpha)
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
		
	case 1.9:
		print("blur")
		



# Exercise 1.3
if 0:
	print("That was a stupid idea")