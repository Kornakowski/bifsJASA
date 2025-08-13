import pywt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy.fft import rfft2, irfft2
from scipy.ndimage import gaussian_filter
import os

## ------------------------------------ INPUT ------------------------------------ ##

folder_name = "ResultsImages/DenoiseComp" # Name of the folder to save the reconstructions

sd_noise = 0.1 # Standard deviation of added noise
thresholds_fourier = [0.1, 0.5, 0.8] # [0.5, 0.7, 1.0] # Threshold values for Fourier denoising
thresholds_wavelet = [0.1, 0.5, 2] # Threshold values for wavelet denoising
wavelet_type = "db4" # Type of wavelet (choose from "db2", "db4", "db8", "sym3", "haar") 
kernel_sds = [1, 2, 4] # Gaussian kernel standard deviations
n_methods = 3 # Number of methods to be applied

N = len(thresholds_fourier) # Number of reconstructions per method


## ------------------------------------ Functions for saving ------------------------------------ ##

os.makedirs(folder_name, exist_ok=True) # Create folder to store the reconstructions
def save_image(image, folder_name, file_name):
    image = (image - image.min()) / (image.max() - image.min()) # normalize
    image = Image.fromarray((image*255).astype(np.uint8)).convert('L')
    image.save(os.path.join(folder_name, file_name), 'PDF')

## ------------------------------------ LOAD DATA ------------------------------------ ##

load_image = Image.open("ExampleImages/Downloaded/mandril_gray.tif") # Load image (mandril)
true_image = np.array(load_image) # convent image to numpy array
image_shape = true_image.shape # save the shape of image
scaled = (true_image.flatten()-min(true_image.flatten()))/(max(true_image.flatten())-min(true_image.flatten())) # scale the image to [0, 1]
true_image = np.reshape(scaled, image_shape) # reshape image to original dimensions
noisy_image = true_image + np.random.normal(0, sd_noise, image_shape) # add Gaussian noise to get the noisy image

# Plot noisy image
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy image')
plt.axis('off')
plt.show()
save_image(noisy_image, folder_name, f'noisy_image.pdf')


## ------------------------------------ APPLY METHODS FOR DENOISING ------------------------------------ ##

# Apply Fourier denoising applying specified threshold
fourier_reconstructions = np.empty((len(thresholds_fourier), image_shape[0], image_shape[1]))
for i in range(len(thresholds_fourier)):
    fourier_signal = rfft2(noisy_image, norm="ortho") # Apply fast Fourier transform (IFFT) to the noisy image
    magnitude = np.sqrt(np.real(fourier_signal)**2 + np.imag(fourier_signal)**2) # Get the magnitude of Fourier transformed image
    phase = np.arctan2(np.imag(fourier_signal), np.real(fourier_signal)) # Get the phase of Fourier transformed image
    thresholded_magnitude = np.where(magnitude < thresholds_fourier[i], 0, magnitude) # Threshold magnitude
    thresholded_fourier_signal = np.vectorize(complex)(thresholded_magnitude * np.cos(phase), thresholded_magnitude * np.sin(phase)) # Get the reconstructed Fourier signal
    fourier_reconstructions[i, :, :] = irfft2(thresholded_fourier_signal, s=image_shape, norm="ortho") # Apply inverse fast Fourier transform (IFFT) to get reconstructed image

# Apply Wavelet denoising applying specified threshold
wavelet_reconstructions = np.empty((len(thresholds_wavelet), image_shape[0], image_shape[1]))
# Perform 2D discrete wavelet transform
coeffs = pywt.wavedec2(noisy_image, wavelet=wavelet_type, level=3) # coeffs[0] is the approximation, coeffs[1:] contains the details (horizontal, vertical, diagonal)
for i in range(len(thresholds_wavelet)):
    # Apply thresholding to detail coefficients
    coeffs_thresh = [coeffs[0]]  # Keep the approximation coefficients unmodified
    for detail_level in coeffs[1:]:
        detail_thresh = tuple(
            np.where(np.abs(detail) > thresholds_wavelet[i], detail, 0) for detail in detail_level
        )
        coeffs_thresh.append(detail_thresh)

    wavelet_reconstructions[i, :, :] = pywt.waverec2(coeffs_thresh, wavelet=wavelet_type) # Reconstruct the image using the thresholded coefficients

# Apply Gaussian filter applying specified Gaussian kernel
gaussian_filter_reconstructions = np.empty((len(thresholds_wavelet), image_shape[0], image_shape[1]))
for i in range(len(kernel_sds)):
    gaussian_filter_reconstructions[i, :, :] = gaussian_filter(noisy_image, sigma=kernel_sds[i])



## ------------------------------------------- PLOT RECONSTUCTIONS ------------------------------------------- ##


# Minimum and maximum values from all the reconstructions (used for plotting the reconstruction in the same scale)
# If that is not needed, the two following rows can be commented out
# vmin = np.min([fourier_reconstructions, wavelet_reconstructions] ) 
# vmax = np.max([fourier_reconstructions, wavelet_reconstructions] )
vmin=0
vmax=1

# Plot reconstructions with different methods
plt.figure(figsize=(10, 10))

# Plot Fourier denoising reconstructions for the chosen thresholds
for i in range(len(thresholds_fourier)):
    plt.subplot(n_methods, N, i+1)
    plt.imshow(fourier_reconstructions[i, :, :], cmap='gray', vmin = vmin, vmax = vmax)
    plt.title(f'Fourier denoising (thresh={thresholds_fourier[i]})')
    plt.axis("off")
    save_image(fourier_reconstructions[i, :, :], folder_name, f'fourier_thresh_{thresholds_fourier[i]}.pdf')

# Plot wavelet denoising reconstructions for the chosen thresholds
for j in range(len(thresholds_wavelet)):
    plt.subplot(n_methods, N, j + N + 1)
    plt.imshow(wavelet_reconstructions[j, :, :], cmap='gray', vmin = vmin, vmax = vmax)
    plt.title(f'Wavelet denoising (thresh={thresholds_wavelet[j]})')
    plt.axis("off")
    save_image(wavelet_reconstructions[j, :, :], folder_name, f'wavelet_thresh_{thresholds_wavelet[j]}.pdf')

# Plot Gaussian filtering reconstructions for the chosen kernel sds
for k in range(len(kernel_sds)):
    plt.subplot(n_methods, N, k + 2*N + 1)
    plt.imshow(gaussian_filter_reconstructions[k], cmap='gray', vmin = vmin, vmax = vmax)
    plt.title(f'Gaussian filter (thresh={kernel_sds[k]})')
    plt.axis("off")
    save_image(gaussian_filter_reconstructions[k], folder_name, f'gaussian_filter_kernel_sds_{kernel_sds[k]}.pdf')

plt.show()

