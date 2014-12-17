class Slicer:

    """Tools to convert a slice number between the various numbering
       schemes. The definition of the various numbering schemes and the
       conversion table can be found in the "Global Positioning System"
       document (VLT-TRE-MUSE-14670-0657).

       All the methods are static and thus there is no need to
       instanciate an object to use this class. E.g.:

           print(Slicer.ccd2sky(4))

           print(Slicer.optical2sky((4, 2)))
       """
    __CCD2SKY__ = dict({1: 9, 2: 8, 3: 1, 4: 10, 5: 7, 6: 2,
                        7: 11, 8: 6, 9: 3, 10: 12, 11: 5, 12: 4,
                        13: 21, 14: 20, 15: 13, 16: 22, 17: 19, 18: 14,
                        19: 23, 20: 18, 21: 15, 22: 24, 23: 17, 24: 16,
                        25: 33, 26: 32, 27: 25, 28: 34, 29: 31, 30: 26,
                        31: 35, 32: 30, 33: 27, 34: 36, 35: 29, 36: 28,
                        37: 45, 38: 44, 39: 37, 40: 46, 41: 43, 42: 38,
                        43: 47, 44: 42, 45: 39, 46: 48, 47: 41, 48: 40})

    __SKY2CCD__ = dict((value, key) for key, value in __CCD2SKY__.iteritems())

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
                           for key, value in __SKY2OPTICAL__.iteritems())

    @staticmethod
    def ccd2sky(s):
        """Convert a slice number from CCD to SKY numbering scheme. Return
        None if the input slice number is invalid.

        :param s: slice number in CCD numbering scheme
        :type s: int
        """
        return Slicer.__CCD2SKY__.get(s)

    @staticmethod
    def sky2ccd(s):
        """Convert a slice number from SKY to CCD numbering scheme. Return
        None if the input slice number is invalid.

        :param s: slice number in SKY numbering scheme
        :type s: int
        """
        return Slicer.__SKY2CCD__.get(s)

    @staticmethod
    def sky2optical(s):
        """Convert a slice number from SKY to OPTICAL numbering scheme. Return
        None if the input slice number is invalid.

        :param s: slice number in SKY numbering scheme
        :type s: int
        """
        return Slicer.__SKY2OPTICAL__.get(s)

    @staticmethod
    def optical2sky(s):
        """Convert a slice number from OPTICAL to SKY numbering scheme. Return
        None if the input slice number is invalid.

        :param s: slice number in OPTICAL numbering scheme
        :type s: tuple of ints
        """
        return Slicer.__OPTICAL2SKY__.get(s)

    @staticmethod
    def optical2ccd(s):
        """Convert a slice number from OPTICAL to CCD numbering scheme. Return
        None if the input slice number is invalid.

        :param s: slice number in OPTICAL numbering scheme
        :type s: tuple of ints
        """

        return Slicer.sky2ccd(Slicer.optical2sky(s))

    @staticmethod
    def ccd2optical(s):
        """Convert a slice number from CCD to OPTICAL numbering scheme. Return
        None if the input slice number is invalid.

        :param s: slice number in CCD numbering scheme
        :type s: int
        """
        return Slicer.sky2optical(Slicer.ccd2sky(s))
