#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 09:32:27 2021.

Functions to extract BIFS posterior estimates.

"""
import numpy as np
import bifsfns as bifs


def emppost(gmrfKmeans, gmrfKsds, magfimgF, argfimgF, knoiseSD, adim=128,
            gmrfPriorScale=1.0):
    """
    GMRF model Conjugate Gaussian empirical k-space estimates.

    Parameters
    ----------
    gmrfKmeans : float numpy.ndarray
        Means of magnitudes of GMRF sims over Fourier space points.
    gmrfKsds : float numpy.ndarray
        SDs of magnitudes of GMRF sims over Fourier space points.
    magfimgF : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimgF : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.
    knoiseSD : float
        Standard deviation estimate of modulus of noise in Fourier space.
    adim : int scalar, optional
        x and y dimensions of brain image matrix. The default is 128.
    gmrfPriorScale : int scalar, optional
        Prior scale value for the GMRF prior. The default is 1.0.

    Returns
    -------
    gmrfEmpiricalRecon : float numpy.ndarray
        Matrix of posterior reconstructed image values -- Gaussian/empirical.

    """
    gmrfKmeans.shape = (adim, adim)
    gmrfKsds.shape = (adim, adim)
    gmrfkMempirical = gmrfPriorScale * gmrfKmeans
    gmrfkSDempirical = gmrfPriorScale * gmrfKsds
    kPostGMRFempirical = bifs.gauss_gauss_post(
        magfimgF, knoiseSD, gmrfkMempirical, gmrfkSDempirical)
    kPostGMRFempirical[0, 0] = magfimgF[0, 0]
    gmrfEmpiricalRecon = np.real(np.fft.ifft2(kPostGMRFempirical
                                              * np.exp(1j * argfimgF),
                                              norm="ortho"))
    return gmrfEmpiricalRecon


def emppost_n(gmrfKmeans, gmrfKsds, magfimgF, argfimgF, knoiseSD, nv=1,
              adim=128, gmrfPriorScale=1.0):
    """
    GMRF model Conjugate Gaussian empirical k-space estimates with prior
    weighted for nv observations.

    Parameters
    ----------
    gmrfKmeans : float numpy.ndarray
        Means of magnitudes of GMRF sims over Fourier space points.
    gmrfKsds : float numpy.ndarray
        SDs of magnitudes of GMRF sims over Fourier space points.
    magfimgF : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimgF : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.
    knoiseSD : float
        Standard deviation estimate of modulus of noise in Fourier space.
    nv : float
        Number of observations from empirical that prior should count for
    adim : int scalar, optional
        x and y dimensions of brain image matrix. The default is 128.
    gmrfPriorScale : int scalar, optional
        Prior scale value for the GMRF prior. The default is 1.0.

    Returns
    -------
    gmrfEmpiricalRecon : float numpy.ndarray
        Matrix of posterior reconstructed image values -- Gaussian/empirical.

    """
    gmrfKmeans.shape = (adim, adim)
    gmrfKsds.shape = (adim, adim)
    kPostGMRFempirical = bifs.gauss_gauss_n_post(
        magfimgF, knoiseSD, gmrfKmeans, gmrfKsds, nval=nv)
    kPostGMRFempirical[0, 0] = magfimgF[0, 0]
    gmrfEmpiricalRecon = np.real(np.fft.ifft2(kPostGMRFempirical
                                              * np.exp(1j * argfimgF),
                                              norm="ortho"))
    return gmrfEmpiricalRecon


def exp2post(gmrfPred, gmrfFittedSD, magfimgF, argfimgF, knoiseSD, adim=128,
             gmrfPriorScale=1.0):
    """
    GMRF model -- Gaussian with exponential of square term model par fn.

    Parameters
    ----------
    gmrfPred : float numpy.ndarray
        Predicted magnitude of modulus given parameter estimates.
    gmrfFittedSD : float numpy.ndarray
        Matrix of fitted SD estimates.
    magfimgF : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimgF : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.
    knoiseSD : float
        Standard deviation estimate of modulus of noise in Fourier space.
    adim : int scalar, optional
        x and y dimensions of brain image matrix. The default is 128.
    gmrfPriorScale : int scalar, optional
        Prior scale value for the GMRF prior. The default is 1.0.

    Returns
    -------
    gmrfExpSqRecon : float numpy.ndarray
        Matrix of posterior reconstructed image -- Gaussian with exp^2 model.

    """
    gmrfPred.shape = (adim, adim)
    gmrfFittedSD.shape = (adim, adim)
    gmrfExpSqFmean = gmrfPriorScale * gmrfPred
    gmrfExpSqFstd = gmrfPriorScale * gmrfFittedSD
    kPostGMRFexpSq = bifs.gauss_gauss_post(
        magfimgF, knoiseSD, gmrfExpSqFmean, gmrfExpSqFstd)
    kPostGMRFexpSq[0, 0] = magfimgF[0, 0]
    gmrfExpSqRecon = np.real(np.fft.ifft2(kPostGMRFexpSq
                                          * np.exp(1j * argfimgF),
                                          norm="ortho"))
    return gmrfExpSqRecon


def exp2flatpost(gmrfPred, gmrfFittedSD, magfimgF, argfimgF, knoiseSD,
                 adim=128, gmrfPriorScale=1.0):
    """
    GMRF model -- exp sq flattened center of Fourier space.

    Parameters
    ----------
    gmrfPred : float numpy.ndarray
        Predicted magnitude of modulus given parameter estimates.
    gmrfFittedSD : float numpy.ndarray
        Matrix of fitted SD estimates.
    magfimgF : float numpy.ndarray
        Matrix of magnitude values at each point in Fourier space.
    argfimgF : float numpy.ndarray
        Matrix of phase/argument values at each point in Fourier space.
    knoiseSD : float
        Standard deviation estimate of modulus of noise in Fourier space.
    adim : int scalar, optional
        x and y dimensions of brain image matrix. The default is 128.
    gmrfPriorScale : int scalar, optional
        Prior scale value for the GMRF prior. The default is 1.0.

    Returns
    -------
    gmrfExpSqFlatRecon : float numpy.ndarray
        Matrix of posterior recon image -- Gaussian with flattened exp^2 model.

    """
    cntr = bifs.indxs(adim, adim)  # to flatten k-space center
    gmrfPred.shape = (adim, adim)
    gmrfFittedSD.shape = (adim, adim)
    gmrfExpSqFmean = gmrfPriorScale * gmrfPred
    gmrfExpSqFstd = gmrfPriorScale * gmrfFittedSD
    gmrfFlattenExpSqMean = bifs.flatten(gmrfExpSqFmean, cntr)
    gmrfFlattenExpSqStd = bifs.flatten(gmrfExpSqFstd, cntr)
    kpostGMRFflatExpSq = bifs.gauss_gauss_post(
        magfimgF, knoiseSD, gmrfFlattenExpSqMean, gmrfFlattenExpSqStd)
    kpostGMRFflatExpSq[0, 0] = magfimgF[0, 0]
    gmrfExpSqFlatRecon = np.real(np.fft.ifft2(kpostGMRFflatExpSq
                                              * np.exp(1j * argfimgF),
                                              norm="ortho"))
    return gmrfExpSqFlatRecon


