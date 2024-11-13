import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def gaussfft(pic, t):
    # Fill in your code here
    # result = ...  

    # 1. Filter based on sampled Gaussian
    [sizex, sizey] = pic.shape
    X, Y = np.meshgrid(
        range(-int(sizex/2), int(sizex/2)), 
        range(-int(sizey/2), int(sizey/2))
        # np.linspace(-sizex/2, sizex/2, sizex), 
        # np.linspace(-sizey/2, sizey/2, sizey)
    )
    gauss = 1/(2*np.pi*t) * np.exp(-1/(2*t) * (X**2+Y**2))

    # 2. Fourier transform image and Gaussian filter
    pic_fft = fft2(pic)
    gauss_fft = fft2(fftshift(gauss))

    # 3. Multiply Fourier transforms
    result_fft = pic_fft * gauss_fft

    # 4. Invert the resulting Fourier transform
    result = ifft2(result_fft)
    # result = fftshift(result)

    return np.real(result)
