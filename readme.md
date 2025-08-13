
Information describing the files contained in the repository.

Repository contains code to reproduce the results in the JASA paper:
Bayesian Image Analysis in Fourier Space -- Kornak, Young, Friedman and Bakas.

Python and package versions used: Python 3.9.23, IPython 8.15.0, numpy 1.26.4, scipy 1.13.1, matplotlib 3.9.2, pillow 11.1.0 fork of PIL 1.1.7, PyWavelets pywt 1.5.0, scikit-learn sklearn 1.6.1.

R and package versions used: R 4.5.1, R-INLA_25.06.22-1 test version

Python and packages were installed using conda 24.11.3.

Spyder 6.0.7 (conda) was used for a Python IDE 
Rstudio 2025.05.0+496 was used with R


Modules in repository
---------------------

The following files are modules that the code imports to run the examples.

bifsfns.py -- the core set of functions for BIFS code 

fitfns.py -- functions to extract fitted parameter functions from simulated MRFs

postfns.py -- functions to extract BIFS posterior estimates

plotfns.py -- functions to aid with plotting/generating maps

simMRIfns.py -- Reads in gray and white map data and generates MNI-simulated brain

mrfconjgrad.py -- functions for MRF conjugate gradients MAP estimation


Python code for examples
------------------------

bifsRunPaperMandrill.py -- code to run example 1 -- Note that to run the t-distributed noise examples, the commented code stating "For t-distributions" in two locations needs to be uncommented and the corresponding Gaussian parts commented. The code needs to be run separately for each number of degrees of freedom in the t-distribution

bifsRunPaperMoon.py -- code to run example 2

bifsRunPaperPirate.py -- code to run example 3

bifsMRFgaussBrain.py -- code to run example 4 -- This requires that inlaGMRFsimCodeForPaper.R be run first to generate the GMRF simulations, which in turn requires R-INLA to be installed. 

simGaussianBellsFT.py -- code to run example 5 -- This is slow due to the simulation and storage in memory of 10,000 images for the empirical distribution

denoising.py -- code for running other denoising methods

Results for all examples after running code can be found in the ResultsImages directory and its sub-directories


R code to simulate GMRFs with R-INLA
------------------------------------

inlaGMRFsimCodeForPaper.R -- code to simulate GMRFs with R-INLA

