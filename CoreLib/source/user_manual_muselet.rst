MUSELET
*******

.. figure:: user_manual_muselet/muselet_logo.jpg
  :align: center

Description
===========

MUSELET (for MUSE Line Emission Tracker) is a simple SExtractor-based python tool to 
detect emission lines in a datacube. It has been developed by Johan Richard (johan.richard@univ-lyon1.fr), 
with help from Johany Martinez (CRAL) and Laure Piqueras (CRAL).

MUSELET uses SExtractor (Bertin & Arnouts 1996, http://www.astromatic.net/software/sextractor) to 
detect line emission in narrow-band images created from the cube. It then merges all detections in 
a single catalog, separating the emission lines linked with continuum sources detected in the white light images 
from the isolated emission lines. It then tries to estimate the redshift from multiple emission lines.

MUSELET takes as an input a MUSE DATACUBE (fits format), and works in 3 steps:

-STEP 1: creation of white light, color and narrow band images.
MUSELET will first create a variance-weighted white light image as well as R,G,B images based on 1/3 of the 
wavelength range each.
The code will then go through the wavelength axis and create one narrow band image at each wavelength plane.
The narrow band image is based on a line-weighted (spectrally) average of 5 wavelength planes in the cube 
(so 5x1.25 Angstroms wide). The continuum is estimated from 2 spectral medians of ~ 25 Angstroms each on the 
blue and red side of the narrow band region. The size of the continuum region can be adjusted with the optionnal 
parameter delta (in number of wavelength planes, default=20).

These narrow band images are created in the "nb/" directory. If not present it will be created.

-STEP 2: 
MUSELET will run SExtractor using the default.sex,default.param,default.conv and default.nnw parameter files 
in the current and nb/ directory. If not present default parameter files are created. 

-STEP 3:
The code will merge all SExtractor catalogs, separate emission lines linked with continuum objects from the rest, 
and produce a list of continuum emission lines (continuum_lines.cat) and isolated emission lines (single_lines.cat).

Then, for each of these catalog, MUSELET will estimate a redshift based on multiple emission lines, and produces 
the corresponding files continuum_lines_z.cat and single_lines_z.cat. Emission lines are merged spatially to the same 
source based on the "radius" parameter (in pixels, default radius=4).

The redshifts are estimated from emission line catalogs emlines (all emission lines) and emlines_small (list of brightest 
emission lines). These files are 2 columnes (name and wavelength) and can be adjusted to one's needs.


Requirements:

- MPDAF (v 1.1.16)

- SExtractor ("sex" binary file in your $PATH).


Tutorials
=========

MUSELET is run through the following commands in mpdaf:

  >>> from mpdaf.sdetect import muselet
  >>> muselet('DATACUBE.fits')

Optionally, one can provide the starting step (2 or 3) in order to 
only redo one part of the script.

  >>> muselet('DATACUBE.fits',step=2) #will assume the narrow-band images are already created

Optionally, one can provide the size of the continuum region to subtract on each side of the narrow-band 
images.

  >>> muselet('DATACUBE.fits',delta=15) #only 15 wavelength planes in continuum estimate

Reference
=========

:func:`mpdaf.sdetect.muselet <mpdaf.sdetect.muselet>` runs MUSELET.
