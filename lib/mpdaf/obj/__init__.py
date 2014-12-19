
"""Copyright (C) 2011 Centre de Recherche Astronomique de Lyon (CRAL)

Submodules:
=========
image:
    manages image object
"""
__LICENSE__ = """
Copyright (C) 2011  Centre de Recherche Astronomique de Lyon (CRAL)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of AURA and its representatives may not be used to
      endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY CRAL ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

"""
__version__ = '1.1.13'
__date__ = '2014/12/17'

"""
Import the different submodules
"""
from coords import WCS
from coords import WaveCoord
from coords import sexa2deg
from coords import deg2sexa
from coords import deg2hms
from coords import hms2deg
from coords import deg2dms
from coords import dms2deg
from spectrum import Spectrum
from spectrum import Gauss1D
from image import Image
from cube import Cube
from cube import CubeDisk
from astropysics_coords import AstropysicsAngularCoordinate
from image import gauss_image
from image import moffat_image
from image import make_image
from image import composite_image
from image import Gauss2D
from image import Moffat2D
from cube import iter_spe
from cube import iter_ima
import plt_zscale
from cubelist import CubeList
