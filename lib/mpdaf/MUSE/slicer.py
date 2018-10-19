# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2013-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2014-2017 Simon Conseil <simon.conseil@univ-lyon1.fr>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


class Slicer(object):

    """Convert slice number between the various numbering schemes.

    The definition of the various numbering schemes and the conversion table
    can be found in the *"Global Positioning System"* document
    (*VLT-TRE-MUSE-14670-0657*).

    All the methods are static and thus there is no need to instanciate an
    object to use this class.

    Examples
    --------

    Convert slice number 4 in CCD numbering to SKY numbering :

    >>> print(Slicer.ccd2sky(4))
    10

    >>> print(Slicer.sky2ccd(10))
    4

    Convert slice number 12 of stack 3 in OPTICAL numbering to SKY numbering:

    >>> print(Slicer.optical2sky((2, 12)))
    25

    >>> print(Slicer.sky2optical(25))
    (2, 12)

    Convert slice number 12 of stack 3 in OPTICAL numbering to CCD numbering:

    >>> print(Slicer.optical2ccd((2, 12)))
    27

    >>> print(Slicer.ccd2optical(27))
    (2, 12)

    """
    __CCD2SKY__ = dict({1: 9, 2: 8, 3: 1, 4: 10, 5: 7, 6: 2,
                        7: 11, 8: 6, 9: 3, 10: 12, 11: 5, 12: 4,
                        13: 21, 14: 20, 15: 13, 16: 22, 17: 19, 18: 14,
                        19: 23, 20: 18, 21: 15, 22: 24, 23: 17, 24: 16,
                        25: 33, 26: 32, 27: 25, 28: 34, 29: 31, 30: 26,
                        31: 35, 32: 30, 33: 27, 34: 36, 35: 29, 36: 28,
                        37: 45, 38: 44, 39: 37, 40: 46, 41: 43, 42: 38,
                        43: 47, 44: 42, 45: 39, 46: 48, 47: 41, 48: 40})

    __SKY2CCD__ = dict((value, key) for key, value in __CCD2SKY__.items())

    __SKY2OPTICAL__ = dict({1: (4, 12), 2: (4, 11), 3: (4, 10),
                            4: (4, 9), 5: (4, 8), 6: (4, 7),
                            7: (4, 6), 8: (4, 5), 9: (4, 4),
                            10: (4, 3), 11: (4, 2), 12: (4, 1),
                            13: (3, 12), 14: (3, 11), 15: (3, 10),
                            16: (3, 9), 17: (3, 8), 18: (3, 7),
                            19: (3, 6), 20: (3, 5), 21: (3, 4),
                            22: (3, 3), 23: (3, 2), 24: (3, 1),
                            25: (2, 12), 26: (2, 11), 27: (2, 10),
                            28: (2, 9), 29: (2, 8), 30: (2, 7),
                            31: (2, 6), 32: (2, 5), 33: (2, 4),
                            34: (2, 3), 35: (2, 2), 36: (2, 1),
                            37: (1, 12), 38: (1, 11), 39: (1, 10),
                            40: (1, 9), 41: (1, 8), 42: (1, 7),
                            43: (1, 6), 44: (1, 5), 45: (1, 4),
                            46: (1, 3), 47: (1, 2), 48: (1, 1)})

    __OPTICAL2SKY__ = dict((value, key)
                           for key, value in __SKY2OPTICAL__.items())

    @staticmethod
    def ccd2sky(s):
        """Convert a slice number from CCD to SKY numbering scheme.

        Return None if the input slice number is invalid.

        Parameters
        ----------
        s : int
            slice number in CCD numbering scheme

        """
        return Slicer.__CCD2SKY__.get(s)

    @staticmethod
    def sky2ccd(s):
        """Convert a slice number from SKY to CCD numbering scheme.

        Return None if the input slice number is invalid.

        Parameters
        ----------
        s : int
            slice number in SKY numbering scheme

        """
        return Slicer.__SKY2CCD__.get(s)

    @staticmethod
    def sky2optical(s):
        """Convert a slice number from SKY to OPTICAL numbering scheme.

        Return None if the input slice number is invalid.

        Parameters
        ----------
        s : int
            slice number in SKY numbering scheme

        """
        return Slicer.__SKY2OPTICAL__.get(s)

    @staticmethod
    def optical2sky(s):
        """Convert a slice number from OPTICAL to SKY numbering scheme.

        Return None if the input slice number is invalid.

        Parameters
        ----------
        s : tuple of int
            slice number in OPTICAL numbering scheme

        """
        return Slicer.__OPTICAL2SKY__.get(s)

    @staticmethod
    def optical2ccd(s):
        """Convert a slice number from OPTICAL to CCD numbering scheme.

        Return None if the input slice number is invalid.

        Parameters
        ----------
        s : tuple of int
            slice number in OPTICAL numbering scheme

        """
        return Slicer.sky2ccd(Slicer.optical2sky(s))

    @staticmethod
    def ccd2optical(s):
        """Convert a slice number from CCD to OPTICAL numbering scheme.

        Return None if the input slice number is invalid.

        Parameters
        ----------
        s : int
            slice number in CCD numbering scheme

        """
        return Slicer.sky2optical(Slicer.ccd2sky(s))
