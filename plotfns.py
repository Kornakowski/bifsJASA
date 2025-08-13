#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:42:30 2022

Functions for displaying sets of images.

"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import bifsfns as bifs


def plotset(imgname, outdir):
    indvl = len(imgname)
    for i in range(indvl):
        im = bifs.c2g(imgname[i])
        img = Image.fromarray(im)
        img.save(outdir + "img" + str(i) + ".pdf")


class Imageset(object):
    """The Imageset object contains a set of images."""

    def __init__(self, imgset, imnames, rdsp=2, cdsp=3, ndsp=6, cmap='gray',
                 fgsz=(10, 7), rescale=True):
        """Initialize variables for methods to display images.

        Parameters
        ----------
        imgset : list(float numpy.ndarray)
            Set of arrays to display as images.
        imnames : list(str)
            Set of titles to be used for the images.
        rdsp : int, optional
            Number of rows of images displayed. The default is 2.
        cdsp : int, optional
            Number of columns of images displayed. The default is 3.
        ndsp : int, optional
            Total number of images displayed. The default is 6.
        cmap : str, optional
            Colormap to be used for displayed images. The default is 'gray'.
        fgsz : int tuple, optional
            Total size of figure area (width, height). The default is (10, 7).
        rescale : bool, optional
            Whether or not to rescale all images to same dynamic range. \
            The default is True.

        Returns
        -------
        None.

        """
        self.imgset = imgset
        self.imnames = imnames
        self.rdsp = rdsp
        self.cdsp = cdsp
        self.ndsp = ndsp
        self.cmap = cmap
        self.rescale = rescale
        self.fgsz = fgsz

    def mplot(self):
        """Display plots.

        Returns
        -------
        None.

        """
        minval = None
        maxval = None
        if self.rescale:
            minval = np.amin(np.asarray(self.imgset))
            maxval = np.amax(np.asarray(self.imgset))
        fig, axes = plt.subplots(self.rdsp, self.cdsp, figsize=self.fgsz)
        cmap = cm.get_cmap(self.cmap)
        normalizer = Normalize(minval, maxval)
        # im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        for i, ax in enumerate(axes.flat):
            im = ax.pcolormesh(self.imgset[i], cmap=cmap, vmin=minval,
                               vmax=maxval)
            ax.set_axis_off()
            ax.imshow(self.imgset[i], cmap=cmap, norm=normalizer)
            ax.set_title(self.imnames[i])
        # fig.colorbar(im, ax=axes.ravel().tolist())
        if self.rescale:
            # fig.colorbar(im, ax=axes.flat)
            # fig.colorbar(im, ax=axes.flat().tolist())
            fig.colorbar(im, ax=axes.ravel().tolist())
        plt.show()


class ImagesetPlots(object):
    """The Imageset object contains a set of images."""

    def __init__(self, imgset, imnames, rescaleVals, rdsp=2, cdsp=3, ndsp=6,
                 cmap='gray', fgsz=(10, 7)):
        """Initialize variables for methods to display images.

        Parameters
        ----------
        imgset : list(float numpy.ndarray)
            Set of arrays to display as images.
        imnames : list(str)
            Set of titles to be used for the images.
        rescaleVals : list(int)
                Image indexes to scale to same dynamic range
        rdsp : int, optional
            Number of rows of images displayed. The default is 2.
        cdsp : int, optional
            Number of columns of images displayed. The default is 3.
        ndsp : int, optional
            Total number of images displayed. The default is 6.
        cmap : str, optional
            Colormap to be used for displayed images. The default is 'gray'.
        fgsz : int tuple, optional
            Total size of figure area (width, height). The default is (10, 7).


        Returns
        -------
        None.

        """
        self.imgset = imgset
        self.imnames = imnames
        self.rdsp = rdsp
        self.cdsp = cdsp
        self.ndsp = ndsp
        self.cmap = cmap
        self.rescaleVals = rescaleVals
        self.fgsz = fgsz

    def mplot(self):
        """Display plots.

        Returns
        -------
        None.

        """
        minval = None
        maxval = None
        imgsetRescale = [self.imgset[i] for i in self.rescaleVals]
        minval = np.amin(np.asarray(imgsetRescale))
        maxval = np.amax(np.asarray(imgsetRescale))
        fig, axes = plt.subplots(self.rdsp, self.cdsp, figsize=self.fgsz)
        cmap = cm.get_cmap(self.cmap)
        normalizer = Normalize(minval, maxval)
        # im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        for i, ax in enumerate(axes.flat):
            if (i in self.rescaleVals):
                im = ax.pcolormesh(self.imgset[i], cmap=cmap, vmin=minval,
                                   vmax=maxval)
                ax.set_axis_off()
                ax.imshow(self.imgset[i], cmap=cmap, norm=normalizer)
                ax.set_title(self.imnames[i])
                # fig.colorbar(im, ax=axes.ravel().tolist())
            else:
                im = ax.pcolormesh(self.imgset[i], cmap=cmap)
                ax.set_axis_off()
                ax.imshow(self.imgset[i], cmap=cmap)
                ax.set_title(self.imnames[i])
                # fig.colorbar(im, ax=axes.ravel().tolist())
        # fig.colorbar(im, ax=axes.flat)
        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.show()


class ImagesetOutPlots(object):
    """The Imageset object contains a set of images."""

    def __init__(self, imgset, imnames, rescaleVals, outdir, rdsp=2, cdsp=3,
                 ndsp=6, cmap='gray', fgsz=(10, 7)):
        """Initialize variables for methods to display images.

        Parameters
        ----------
        imgset : list(float numpy.ndarray)
            Set of arrays to display as images.
        imnames : list(str)
            Set of titles to be used for the images.
        rescaleVals : list(int)
                Image indexes to scale to same dynamic range
        rdsp : int, optional
            Number of rows of images displayed. The default is 2.
        cdsp : int, optional
            Number of columns of images displayed. The default is 3.
        ndsp : int, optional
            Total number of images displayed. The default is 6.
        cmap : str, optional
            Colormap to be used for displayed images. The default is 'gray'.
        fgsz : int tuple, optional
            Total size of figure area (width, height). The default is (10, 7).


        Returns
        -------
        None.

        """
        self.imgset = imgset
        self.imnames = imnames
        self.rdsp = rdsp
        self.cdsp = cdsp
        self.ndsp = ndsp
        self.cmap = cmap
        self.rescaleVals = rescaleVals
        self.fgsz = fgsz
        self.outdir = outdir

    def mplot(self):
        """Display plots.

        Returns
        -------
        None.

        """
        minval = None
        maxval = None
        indvl = len(self.imnames)
        imgsetRescale = [self.imgset[i] for i in self.rescaleVals]
        minval = np.amin(np.asarray(imgsetRescale))
        maxval = np.amax(np.asarray(imgsetRescale))
        fig, axes = plt.subplots(self.rdsp, self.cdsp, figsize=self.fgsz)
        cmap = cm.get_cmap(self.cmap)
        normalizer = Normalize(minval, maxval)
        # im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        for i, ax in enumerate(axes.flat):
            if (i in self.rescaleVals):
                im = ax.pcolormesh(self.imgset[i], cmap=cmap, vmin=minval,
                                   vmax=maxval)
                ax.set_axis_off()
                ax.imshow(self.imgset[i], cmap=cmap, norm=normalizer)
                ax.set_title(self.imnames[i])
            else:
                im = ax.pcolormesh(self.imgset[i], cmap=cmap)
                ax.set_axis_off()
                ax.imshow(self.imgset[i], cmap=cmap)
                ax.set_title(self.imnames[i])

        for i in range(indvl):
            if (i in self.rescaleVals):
                im = bifs.c2gAdjust(self.imgset[i], minval, maxval)
            else:
                im = bifs.c2g(self.imgset[i])
            img = Image.fromarray(im)
            img.save(self.outdir + "img" + self.imnames[i] + ".pdf")
