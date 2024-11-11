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

match EXERCISE:
	case 1.3:
		# QUESTION 1:
		# Repeat this exercise with the coordinates p and q set to (5, 9), (9, 5), (17, 9), (17, 121),
		# (5, 1) and (125, 1) respectively. What do you observe?
		for u, v in [(5, 9), (9, 5), (17, 9), (17, 121), (5, 1), (125, 1)]:
			fftwave(u, v)

		# QUESTION 3:
		# How large is the amplitude? Write down the expression derived from Equation (4) in
		# these notes. Complement the code (variable amplitude) accordingly

# Exercise 1.3
if 0:
	print("That was a stupid idea")