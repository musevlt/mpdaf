# Copyright 2008 Erik Tollerud
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   Change by LP for MPDAF
#


import numpy as np
from math import pi


class AstropysicsAngularCoordinate(object):

    """A class representing an angular value.

    Arithmetic operators can be applied to the coordinate, and will be applied
    directly to the numerical value in radians.  For + and -, two angular
    coordinates may be used, although for -, an AngularSeparation object will
    be returned.
    """
    import re as _re
    __slots__ = ('_decval', '_range')

    # this disturbingly complex RE matches anything that looks
    # like a standard sexigesimal or similar string
    __acregex = _re.compile(r'(?:([+-])?(\d+(?:[.]\d*)?)(hours|h|degrees|d|radians|rads|rad|r| |:(?=\d+:\d+[.]?\d*$)))?(?:(\d+(?:[.]\d*)?)(m|\'|[:]| ))?(?:(\d+(?:[.]\d*)?)(s|"|$))?$')
    # and this one matches all things that look like raw numbers
    __decregex = _re.compile(r'[+-]?\d+([.]\d*)?$')

    def __init__(self, inpt=None, sghms=None, range=None, radians=False):
        """
        The input parser is very adaptable, and can be in any of the following
        forms for `inpt`:

        * A float value
            if `radians` is True, this will be interpreted as decimal radians,
            otherwise, it is in degrees.
        * An :class:`AstropysicsAngularCoordinate` object
            A copy of the input object will be created.
        * None
            The default of 0 will be used.
        * A 3-tuple
            If `sghms` is True, the tuple will be interpreted as
            (hours,min,sec), otherwise, (degrees,min,sec).
        * A string of the form ##.##
            If `radians` is True, this will be cast to a float and used as
            decimal radians, otherwise, it is in degrees.
        * A string of the form ##.##d or ##.##degrees
            The numerical part will be cast to a float and used as degrees.
        * A string of the form ##.##h or ##.##hours
            The numerical part will be cast to a float and used as hours.
        * A string of the form ##.##radians,##.##rads, or ##.##r
            The numerical part will be cast to a float and used as radians.
        * A string of the form (+/-)##h##m##.##s
            The numerical parts will be treated as hours,minutes, and seconds.
        * A string of the form (+/-)##d##m##.##s or (+/-)##d##'##.##"
            The numerical parts will be treated as degrees,minutes,
            and seconds.
        * A string of the form (+/-)##:##:##.## or (+/-)## ## ##.##
            Sexigesimal form. If `sghms` is None the presence of a a + or -
            signidicates that it should be interpreted as degrees, minutes,
            and seconds. If the sign is absent, the numerical portions will be
            treated as hours,min,sec. thewise, if `sghms` evaluates to True,
            thenumerical parts will be treated as hours,minutes, and seconds,
            and if `sghms` evaluates to False, degrees,minutes, and seconds.

Parameters
----------
inpt    : float
          The coordinate value -- valid forms are described above.
sghms   : boolean       
          If True, ambiguous sexigesimal inputs should be hours, minutes,
          and seconds instead of degrees,arcmin, and arcsec
range   : (float, float)
          Sets the valid range of coordinates.  Either a
          2-sequence (lowerdegrees,upperdegrees) or None (for no limit)
radians : boolean
          If True, ambiguous inputs are treated as radians rather than
        degrees.

        **Examples**

        >>> from math import pi
        >>> ac = AstropysicsAngularCoordinate(2.5)
        >>> print ac
        +2d30'00.00"
        >>> print AstropysicsAngularCoordinate(ac)
        +2d30'00.00"
        >>> print AstropysicsAngularCoordinate(pi,radians=True)
        +180d00.00"
        >>> print AstropysicsAngularCoordinate('1.1')
        +1d6'00.00"
        >>> print AstropysicsAngularCoordinate('1.1',radians=True)
        +63d1'31.29"
        >>> print AstropysicsAngularCoordinate('12d25m12.5s')
        +12d25'12.50"
        >>> print AstropysicsAngularCoordinate('3:30:30',sghms=True)
        +52d37'30.00"
        >>> print AstropysicsAngularCoordinate('3:30:30',sghms=False)
        +3d30'30.00"
        >>> print AstropysicsAngularCoordinate('-3:30:30',sghms=None)
        -3d30'30.00"
        >>> print AstropysicsAngularCoordinate('+3:30:30',sghms=None)
        +3d30'30.00"
        >>> print AstropysicsAngularCoordinate('3:30:30',sghms=None)
        +52d37'30.00"

        """
        from operator import isSequenceType

        self._range = None

        if isinstance(inpt, AstropysicsAngularCoordinate):
            self._decval = inpt._decval
            self._range = inpt._range
            return
        elif inpt is None:
            self._decval = 0
        elif isinstance(inpt, basestring):
            sinpt = inpt.strip()

            decm = self.__decregex.match(sinpt)
            if decm:
                if radians:
                    self.radians = float(decm.group(0))
                else:
                    self.degrees = float(decm.group(0))
            else:
                acm = self.__acregex.match(sinpt)
                if acm:
                    sgn, dec1, mark1, dec2, mark2, dec3, mark3 = \
                        acm.group(1, 2, 3, 4, 5, 6, 7)
                    val = (0 if dec1 is None else float(dec1)) + \
                          (0 if dec2 is None else float(dec2) / 60) + \
                          (0 if dec3 is None else float(dec3) / 3600)
                    if sgn == '-':
                        val *= -1
                    if mark1 == ':' or mark1 == ' ':
                        if sghms is None:
                            if sgn is None:
                                self.hours = val
                            else:  # '+' or '-'
                                self.degrees = val
                        elif sghms:
                            self.hours = val
                        else:
                            self.degrees = val
                    elif mark1 == 'hours' or mark1 == 'h':
                        self.hours = val
                    elif mark1 == 'degrees' or mark1 == 'd':
                        self.degrees = val
                    elif mark1 == 'radians' or mark1 == 'rad' or \
                            mark1 == 'rads' or mark1 == 'r':
                        self.radians = val
                    else:
                        try:
                            if radians:
                                self.radians = float(val)
                            else:
                                self.degrees = float(val)
                        except ValueError:
                            raise ValueError('invalid string input '
                                             'for AstropysicsAngularCoordinate')
                else:
                    raise ValueError('Invalid string input for '
                                     'AstropysicsAngularCoordinate: ' + inpt)

        elif isSequenceType(inpt) and len(inpt) == 3:
            if sghms:
                self.hrsminsec = inpt
            else:
                self.degminsec = inpt
        elif radians:
            self._decval = float(inpt)
        else:
            from math import radians
            self._decval = radians(inpt)

        self.range = range

    def _setDegminsec(self, dms):
        if not hasattr(dms, '__iter__') or len(dms) != 3:
            raise ValueError('Must set degminsec as a length-3 iterator')
        self.degrees = abs(dms[0]) + abs(dms[1]) / 60. + abs(dms[2]) / 3600.
        if dms[0] < 0:
            self._decval *= -1

    def _getDegminsec(self):
        fulldeg = abs(self.degrees)
        deg = int(fulldeg)
        fracpart = fulldeg - deg
        min = int(fracpart * 60.)
        sec = fracpart * 3600. - min * 60.
        return -deg if self.degrees < 0 else deg, min, sec

    degminsec = property(_getDegminsec, _setDegminsec, doc="""
    The value of this :class:`AstropysicsAngularCoordinate` as an (degrees,
    minutes,seconds) tuple, with degrees and minutes as integers and seconds
     as a float.
    """)
    dms = degminsec

    def _setHrsminsec(self, dms):
        if not hasattr(dms, '__iter__') or len(dms) != 3:
            raise ValueError('Must set hrsminsec as a length-3 iterator')
        self.degrees = 15 * (dms[0] + dms[1] / 60. + dms[2] / 3600.)

    def _getHrsminsec(self):
        factorized = self.degrees / 15.
        hrs = int(factorized)
        mspart = factorized - hrs
        min = int(mspart * 60.)
        sec = mspart * 3600. - min * 60.
        return hrs, min, sec

    hrsminsec = property(_getHrsminsec, _setHrsminsec, doc="""
    The value of this :class:`AstropysicsAngularCoordinate` as an (hours,minutes,seconds)
    tuple, with hours and minutes as integers and seconds as a float.
    """)
    hms = hrsminsec

    def _setDecdeg(self, deg):
        rads = deg * pi / 180.
        if self.range is not None:
            rads = self._checkRange(rads)
        self._decval = rads

    def _getDecdeg(self):
        return self._decval * 180 / pi
    degrees = property(_getDecdeg, _setDecdeg, doc="""
    The value of this :class:`AstropysicsAngularCoordinate` in decimal degrees.
    """)
    d = degrees

    def _setRad(self, rads):
        if self.range is not None:
            rads = self._checkRange(rads)
        self._decval = rads

    def _getRad(self):
        return self._decval
    radians = property(_getRad, _setRad, doc="""
    The value of this :class:`AstropysicsAngularCoordinate` in decimal radians.
    """)
    r = radians

    def _setDechr(self, hr):
        rads = hr * pi / 12
        if self.range is not None:
            rads = self._checkRange(rads)
        self._decval = rads

    def _getDechr(self):
        return self._decval * 12 / pi
    hours = property(_getDechr, _setDechr, doc="""
    The value of this :class:`AstropysicsAngularCoordinate` in decimal hours.
    """)
    h = hours

    def _checkRange(self, rads):
        """
        Checks if the input value is in range - returns the new value,
        or raises
        a :exc:`ValueError`.
        """
        if self._range is not None:
            low, up, cycle = self._range
            if cycle is None:
                if low <= rads <= up:
                    return rads
                else:
                    raise ValueError('Attempted to set angular coordinate '
                                     'outside range')
            else:
                if cycle > 0:
                    # this means use "triangle wave" pattern
                    # with the given quarter-period
                    from math import sin
                    from math import asin
                    offset = low / (low - up) - 0.5
                    return (up - low) * \
                        (asin(sin(pi * (2 * rads / cycle + offset)))
                         / pi + 0.5) + low
                else:
                    return (rads - low) % (up - low) + low
        else:
            return rads

    def _setRange(self, newrng):
        oldrange = self._range
        try:
            if newrng is None:
                self._range = None
            else:
                from math import radians
                newrng = tuple(newrng)
                if len(newrng) == 2:
                    if newrng[1] - newrng[0] == 360:
                        newrng = (newrng[0], newrng[1], 0)
                    else:
                        newrng = (newrng[0], newrng[1], None)
                elif len(newrng) == 3:
                    pass
                else:
                    raise TypeError('range is not a 2 or 3-sequence')
                if newrng[0] > newrng[1]:
                    raise ValueError('lower edge of range is not <= upper')

                newrng = (radians(newrng[0]), radians(newrng[1]),
                          None if newrng[2] is None else radians(newrng[2]))
            self._range = newrng
            self._decval = self._checkRange(self._decval)
        except ValueError, e:
            self._range = oldrange
            if e.args[0] == 'lower edge of range is not <= upper':
                raise e
            else:
                raise ValueError('Attempted to set range '
                                 'when value is out of range')

    def _getRange(self):
        if self._range is None:
            return None
        else:
            from math import degrees
            if self._range[2] is None:
                return degrees(self._range[0]), degrees(self._range[1])
            else:
                return degrees(self._range[0]), degrees(self._range[1]), \
                    degrees(self._range[2])
    range = property(_getRange, _setRange, doc="""
    The acceptable range of angles for this :class:`AstropysicsAngularCoordinate`.
    This can be set as a 2-sequence (lower,upper), or as a 3-sequence
    (lower,upper,cycle), where cycle can be :

        * 0: Angle values are coerced to lie in the range (default for
          2-sequence if upper-lower is 360 degrees)
        * None: A :exc:`ValueError` will be raised if out-of-range (default for
          2-sequence otherwise)
        * A positive scalar: Values are coerced in a triangle wave scheme, with
          the scalar specifying the period. e.g. for the latitude, (-90,90,360)
          would give the correct behavior)
    """)

    def __str__(self):
        return self.getDmsStr(sep=('d', "'", '"'))

    def __eq__(self, other):
        if hasattr(other, '_decval'):
            return self._decval == other._decval
        else:
            return self._decval == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if hasattr(other, '_decval'):
            res = self.__class__()
            res._decval = self._decval + other._decval
        else:
            res = self.__class__()
            res._decval = self._decval + other
        return res

    def __sub__(self, other):
        if hasattr(other, '_decval'):
            res = self.__class__()
            res._decval = self._decval - other._decval
        else:
            res = self.__class__()
            res._decval = self._decval - other
        return res

    def __mul__(self, other):
        res = self.__class__()
        res._decval = self._decval * other
        return res

    def __div__(self, other):
        res = self.__class__()
        res._decval = self._decval / other
        return res

    def __truediv__(self, other):
        res = self.__class__()
        res._decval = self._decval // other
        return res

    def __pow__(self, other):
        res = self.__class__()
        res._decval = self._decval ** other
        return res

    def __float__(self):
        return self.degrees

    def getDmsStr(self, secform='%05.2f', sep=(unichr(176), "'", '"'),
                  sign=True, canonical=False):
        """Generates the string representation of this
        AstropysicsAngularCoordinate as degrees, arcminutes, and arcseconds.

        :param secform: a formatter for the seconds
        :type secform: string
        :param sep:
            The seperator between components - defaults to degree sign, ' and "
            symbols.
        :type sep: string or 3-tuple of strings
        :param sign: Forces sign to be present before degree component.
        :type sign: boolean
        :param canonical: forces [+/-]dd:mm:ss.ss , overriding other arguments

        :returns: String representation of this object.
        """
        d, m, s = self.degminsec

        if canonical:
            sgn = '' if self._decval < 0 else '+'
            return '%s%02.i:%02.i:%05.2f' % (sgn, d, m, s)

        d, m = str(d), str(m)

        s = secform % s

        if isinstance(sep, basestring):
            if sep == 'dms':
                sep = ('d', 'm', 's')
            sep = (sep, sep)

        tojoin = []

        if sign and self._decval >= 0:
            tojoin.append('+')

        if d is not '0':
            tojoin.append(d)
            tojoin.append(sep[0])

        if m is not '0':
            tojoin.append(m)
            tojoin.append(sep[1])

        tojoin.append(s)
        if len(sep) > 2:
            tojoin.append(sep[2])

        return ''.join(tojoin)

    def getHmsStr(self, secform=None, sep=('h', 'm', 's'), canonical=False):
        """gets the string representation of this AstropysicsAngularCoordinate
        as hours, minutes, and seconds.

        secform is the formatter for the seconds component

        sep is the seperator between components -
        defaults to h, m, and s

        canonical forces [+/-]dd:mm:ss.ss , overriding other arguments

        Generates the string representation of
        this AstropysicsAngularCoordinate as hours,
        minutes, and seconds.

        Parameters
        ----------
        secform   : string
                    a formatter for the seconds component
        sep       : string or 3-tuple of strings
                    The seperator between components - defaults to 'h', 'm', and 's'.
        canonical : boolean
                    Forces [+/-]dd:mm:ss.ss , overriding other arguments

        :returns: String representation of this object.
        """

        h, m, s = self.hrsminsec

        if canonical:
            return '%02.i:%02.i:%06.3f' % (h, m, s)

        h, m = str(h), str(m)
        if secform is None:
            s = str(s)
        else:
            s = secform % s

        if isinstance(sep, basestring):
            if sep == 'hms':
                sep = ('h', 'm', 's')
            sep = (sep, sep)

        tojoin = []

        tojoin.append(h)
        tojoin.append(sep[0])

        tojoin.append(m)
        tojoin.append(sep[1])

        tojoin.append(s)
        if len(sep) > 2:
            tojoin.append(sep[2])

        return ''.join(tojoin)
