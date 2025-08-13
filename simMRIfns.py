#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jun  4 12:55:50 2021.

Functions to read in the brain tissue maps and simulate the MRI brain

"""
import numpy as np
import bifsfns as bifs

gfl = ("ExampleImages/MNIdata2D/gmask.dat")
wfl = ("ExampleImages/MNIdata2D/wmask.dat")


def simbrainMRI(adim=128, imsc=10.0, gi=2.0, wi=1.0, sdi=0.25, gfile=gfl,
                wfile=wfl, seedval=0):
    """Create simulation of the brain plus noise etc.

    Parameters
    ----------
    adim : int scalar, optional
        x and y dimensions of brain image matrix. The default is 128.
    imsc : float, optional
        Image scale constant for brain simulation. The default is 10.0.
    gi : float, optional
        Relative intensity for gray matter. The default is 2.0.
    wi : float, optional
        Relative intensity for white matter. The default is 1.0.
    sdi : float, optional
        Relative intensity for noise. The default is 0.25.
    gfile : string, optional
        Filename for the gray matter mask. The default is gfl.
    wfile : string, optional
        Filename for the white matter mask. The default is wfl.
    seedval : int, optional
        Value for the random seed. The default is 0.

    Returns
    -------
    noiseSD : float
        Standard deviation of noise in image space.
    knoiseSD : float
        Estimated standard deviation of modulus of noise in Fourier space.
    imgPlusNoise : float numpy.ndarray
        Output image including noise.
    cleanImage : float numpy.ndarray
        Output image without added noise.
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    invkdst : float numpy.ndarray
        Matrix of inverse of distances from origin in FS (i.e. 1/kdst).
    magfimgF : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimgF : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.
    knoiseSDEst : float
        Rayleigh-based standard deviation of modulus of noise in Fourier space.
    gmsk : float numpy.ndarray
        Matrix of gray mask
    wmsk : float numpy.ndarray
        Matrix of white mask
    """
    with open(gfile, 'rb') as file:
        dat = np.fromfile(file, dtype=np.int32)
        gmsk = np.reshape(dat, [adim, adim])

    with open(wfile, 'rb') as file:
        dat = np.fromfile(file, dtype=np.int32)
        wmsk = np.reshape(dat, [adim, adim])

    totalPixels = adim * adim
    gmsk.astype(np.float64)
    wmsk.astype(np.float64)
    gmsk = np.transpose(gmsk[:, ::-1])
    wmsk = np.transpose(wmsk[:, ::-1])
    grayIntensity = gi * imsc
    whiteIntensity = wi * imsc
    MRmap = grayIntensity * gmsk + whiteIntensity * wmsk
    cleanImage = np.copy(MRmap)
    cleanImage.astype(np.float64)
    shapeImage = np.asarray(cleanImage.shape)
    np.random.seed(seedval)
    noiseSD = sdi * imsc
    noise = np.reshape(np.random.normal(0.0, noiseSD,
                                        cleanImage.size), shapeImage)
    imgPlusNoise = cleanImage + noise
    fftimgF = np.fft.fft2(imgPlusNoise, norm="ortho")  # full 2D fft
    magfimgF = np.abs(fftimgF)
    argfimgF = np.angle(fftimgF)  # Extract corresponding phase image
    kdst = bifs.kdist2D(cleanImage.shape[0], cleanImage.shape[1])
    invkdst = 1/kdst
    fftNoise = np.fft.fft2(noise, norm="ortho")
    knoiseSDest = np.std(np.abs(fftNoise), ddof=1)
    knoiseSD = np.sqrt(noiseSD**2 * (1 - np.pi/4))
    kdst.shape = (totalPixels, )
    return noiseSD, knoiseSD, imgPlusNoise, cleanImage, kdst, invkdst, \
        magfimgF, argfimgF, knoiseSDest, gmsk, wmsk


def simFreq(adim=128, mag1=2, mag2=1, mag3=3, freq1=0.5, freq2=0.8, freq3=0.1,
            phase1=0.5, phase2=1.0, phase3=0.1, sdi=0.25, seedval=0):
    """Create simulation of the brain plus noise plus drift over space. etc.

    Parameters
    ----------
    adim : int scalar, optional
        x and y dimensions of brain image matrix. The default is 128.
    mag1 : float scalar
        Magnitude of first frequency. Default = 2.
    mag2 : float scalar
        Magnitude of second frequency. Default = 1.
    mag3 : float scalar
        Magnitude of third frequency. Default = 3.
    freq1 : float scalar
        First frequency value. Default = 0.5.
    freq2 : float scalar
        Second frequency value. Default = 0.8.
    freq3 : float scalar
        Third frequency value. Default = 0.1.
    phase1 : float scalar
        First phase value. Default = 0.5.
    phase2 : float scalar
        Second phase value. Default = 1.0.
    phase3 : float scalar
        Third phase value. Default = 0.1.
    sdi : float scalar
        Noise SD in image space. Default is 0.25.
    seedval : int, optional
        Value for the random seed. The default is 0.

    Returns
    -------
    noiseSD : float
        Standard deviation of noise in image space.
    knoiseSD : float
        Estimated standard deviation of modulus of noise in Fourier space.
    imgPlusNoise : float numpy.ndarray
        Output image including noise.
    cleanImage : float numpy.ndarray
        Output image without added noise.
    kdst : float numpy.ndarray
        Matrix of distances from origin in Fourier space.
    invkdst : float numpy.ndarray
        Matrix of inverse of distances from origin in FS (i.e. 1/kdst).
    magfimgF : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimgF : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.
    knoiseSDEst : float
        Rayleigh-based standard deviation of modulus of noise in Fourier space.

    """

    totalPixels = adim * adim
    cleanImage = np.zeros([adim, adim])
    for i in range(adim):
        for j in range(adim):
            cleanImage[i, j] = mag1 * np.sin(i*freq1*2*np.pi - phase1) \
                + mag2 * np.sin(j*freq2*2*np.pi - phase2) \
                + mag3 * np.sin((i+j)*freq3*2*np.pi - phase3)
    cleanImage.astype(np.float64)
    shapeImage = np.asarray(cleanImage.shape)
    np.random.seed(seedval)
    noiseSD = sdi
    noise = np.reshape(np.random.normal(0.0, noiseSD,
                                        cleanImage.size), shapeImage)
    imgPlusNoise = cleanImage + noise
    fftimgF = np.fft.fft2(imgPlusNoise, norm="ortho")  # full 2D fft
    magfimgF = np.abs(fftimgF)
    argfimgF = np.angle(fftimgF)  # Extract corresponding phase image
    kdst = bifs.kdist2D(cleanImage.shape[0], cleanImage.shape[1])
    invkdst = 1/kdst
    fftNoise = np.fft.fft2(noise, norm="ortho")
    knoiseSDest = np.std(np.abs(fftNoise), ddof=1)
    knoiseSD = np.sqrt(noiseSD**2 * (1 - np.pi/4))
    kdst.shape = (totalPixels, )
    return noiseSD, knoiseSD, imgPlusNoise, cleanImage, kdst, invkdst, \
        magfimgF, argfimgF, knoiseSDest


def manipGmrfSim(cleanImage, gmrfl, adim=128, nsamps=1000):
    """Read Gaussian MRF simulations and performing manipulations thereof.

    A) read GMRF sims; B) take FFT of each sim; C) convert to magnitude and
    phase maps; D) calculate mean and sd of magnitude maps across sims.

    Parameters
    ----------
    cleanImage : float numpy.ndarray
        Noise free version of image of interest.
    gmrfl : string
        Filename with the set of GMRF simulations.
    adim : int, optional
        x and y dimensions of brain image matrix. The default is 128.
    nsamps : int, optional
        Number of samples of GMRF used. The default is 1000.

    Returns
    -------
    gmrfmat : float numpy.ndarray
        adim x adim simulations times nsamps = adim x adim x nsamps array.
    gmrfsd : float
        Estimated marginal SD of zero mean GMRF.
    gmrfKmeans : float numpy.ndarray
        Means of magnitudes over Fourier space points.
    gmrfKsds : float numpy.ndarray
        SDs of magnitudes over Fourier space points.
    gmrfFFT : complex numpy.ndarray
        FFT of gmrfmat.
    gmrfModFFT : float numpy.ndarray
        Absolute values of FFT of gmrfmat.
    gmrfModFFTsq : float numpy.ndarray
        Absolute values of FFT of gmrfmat squared.
    gmrfMeanModFFTsq : float numpy.ndarray
        Mean over samples of absolute values of FFT of gmrfmat squared.
    gmrfSDevModFFTsq : float numpy.ndarray
        Std dev over samples of absolute values of FFT of gmrfmat squared.

    """
    with open(gmrfl, 'rb') as file:
        gmrfdat = np.fromfile(file, dtype=np.float64)

    gmrfmat = np.reshape(gmrfdat, [nsamps, adim, adim])
    for i in range(nsamps):
        gmrfmat[i, :, :] = gmrfmat[i, :, :] - np.mean(gmrfmat[i, :, :])

    gmrfsd = np.std(gmrfmat, ddof=1)
    # Generating FFT based simulations (marginally indep priors in k-space)
    gmrfFFT = np.zeros(gmrfmat.shape, dtype=complex)
    for i in range(nsamps):
        gmrfFFT[i, :, :] = np.fft.fft2(gmrfmat[i, :, :], norm="ortho")

    gmrfModFFT = np.abs(gmrfFFT)
    gmrfModFFTsq = np.real(gmrfFFT * np.conj(gmrfFFT))
    gmrfKmeans = np.mean(gmrfModFFT, axis=0)
    gmrfMeanModFFTsq = np.mean(gmrfModFFTsq, axis=0)
    gmrfKsds = np.std(gmrfModFFT, axis=0, ddof=1)
    gmrfSDevModFFTsq = np.std(gmrfModFFTsq, axis=0)
    totalPixels = adim * adim
    gmrfKmeans.shape = (totalPixels, )
    gmrfMeanModFFTsq.shape = (totalPixels, )
    gmrfKsds.shape = (totalPixels, )
    gmrfSDevModFFTsq.shape = (totalPixels, )
    return gmrfmat, gmrfsd, gmrfKmeans, gmrfKsds, gmrfFFT, gmrfModFFT, \
        gmrfModFFTsq, gmrfMeanModFFTsq, gmrfSDevModFFTsq


