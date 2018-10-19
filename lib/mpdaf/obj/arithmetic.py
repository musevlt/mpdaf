# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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

import numpy as np
from numpy import ma

from .data import DataArray
from .objs import UnitMaskedArray, UnitArray


# Docstring templates for add, subtract, multiply, divide methods.
_arit_doc = """
    Performs {name} by evaluating ``self`` {op} ``operand``.

    Operation can be performed with a scalar number, a Numpy ndarray or masked
    array, or a MPDAF object. The dimensions must be equal, or, if ``self`` and
    ``operand`` have compatible shapes, they will be broadcasted together. So
    it is possible to perfom an operation between a `~mpdaf.obj.Cube` and an
    a `~mpdaf.obj.Image` or a `~mpdaf.obj.Spectrum`. For MPDAF objects, they
    must also have compatible coordinates (world and wavelength).

    Parameters
    ----------
    operand : int, float, ndarray or `DataArray`
        The second operand in the operation.

    Returns
    -------
    result : `~DataArray`
        The resulting object.

    """


def _check_compatible_coordinates(a, b):
    if a.wave is not None and b.wave is not None and \
            not a.wave.isEqual(b.wave):
        raise ValueError('Operation forbidden for cubes with different world '
                         'coordinates in spectral direction')

    if a.wcs is not None and b.wcs is not None and \
            not a.wcs.isEqual(b.wcs):
        raise ValueError('Operation forbidden for cubes with different world '
                         'coordinates in spatial directions')


def _check_compatible_shapes(a, b, dims=slice(None)):
    if not np.array_equal(a.shape[dims], b.shape[dims]):
        raise ValueError('Operation forbidden for arrays with different '
                         'shapes')


def _arithmetic_data(operation, a, b, newshape=None):
    if a.unit != b.unit:
        data = UnitMaskedArray(b.data, b.unit, a.unit)
    else:
        data = b.data
    if newshape is not None:
        data = data.reshape(newshape)
    return operation(a.data, data)


def _arithmetic_var(operation, a, b, newshape=None):
    if a._var is None and b._var is None:
        return None

    if b._var is not None:
        if a.unit != b.unit:
            var = UnitArray(b._var, b.unit**2, a.unit**2)
        else:
            var = b._var

        if newshape is not None:
            var = var.reshape(newshape)

    if operation in (ma.add, ma.subtract):
        if b.var is None:
            return a.var
        elif a.var is None:
            return np.broadcast_to(var, a.shape)
        else:
            return a.var + var
    elif operation in (ma.multiply, ma.divide):
        b_data = b._data.reshape(newshape)
        if a._var is None:
            var = var * a._data * a._data
        elif b._var is None:
            var = a._var * b_data * b_data
        else:
            var = var * a._data * a._data + a._var * b_data * b_data

        if operation is ma.divide:
            var /= (b_data ** 4)
        return var


def _arithmetic(operation, a, b):
    if a.ndim < b.ndim:
        if operation == ma.subtract:
            return -1 * _arithmetic(operation, b, a)
        elif operation == ma.divide:
            return 1 / _arithmetic(operation, b, a)
        else:
            return _arithmetic(operation, b, a)

    _check_compatible_coordinates(a, b)

    if a.ndim == 3 and b.ndim == 1:  # cube + spectrum
        _check_compatible_shapes(a, b, dims=0)
        newshape = (-1, 1, 1)
    elif a.ndim == 3 and b.ndim == 2:  # cube + image
        _check_compatible_shapes(a, b, dims=slice(-1, -3, -1))
        newshape = (1, ) + b.shape
    elif a.ndim == 2 and b.ndim == 1:  # image + spectrum
        from .cube import Cube
        var = np.expand_dims(a.var, axis=0) if a.var is not None else None
        a = Cube.new_from_obj(a, data=np.expand_dims(a.data, axis=0), var=var)
        a.wave = b.wave.copy()
        newshape = (-1, 1, 1)
    else:
        _check_compatible_shapes(a, b)
        newshape = None

    return a.__class__.new_from_obj(
        a, copy=False,
        data=_arithmetic_data(operation, a, b, newshape=newshape),
        var=_arithmetic_var(operation, a, b, newshape=newshape)
    )


class ArithmeticMixin(object):

    def __add__(self, other):
        if not isinstance(other, DataArray):
            return self.__class__.new_from_obj(
                self, data=self._data + other, copy=True)
        else:
            return _arithmetic(ma.add, self, other)

    def __sub__(self, other):
        if not isinstance(other, DataArray):
            return self.__class__.new_from_obj(
                self, data=self._data - other, copy=True)
        else:
            return _arithmetic(ma.subtract, self, other)

    def __rsub__(self, other):
        if not isinstance(other, DataArray):
            return self.__class__.new_from_obj(
                self, data=other - self._data, copy=True)
        # else:
        #     if other is a DataArray, it is already handled by __sub__

    def __mul__(self, other):
        if not isinstance(other, DataArray):
            res = self.__class__.new_from_obj(
                self, data=self._data * other, copy=True)
            if self._var is not None:
                res._var *= other ** 2
            return res
        else:
            return _arithmetic(ma.multiply, self, other)

    def __div__(self, other):
        if not isinstance(other, DataArray):
            res = self.__class__.new_from_obj(
                self, data=self._data / other, copy=True)
            if self._var is not None:
                res._var /= other ** 2
            return res
        else:
            return _arithmetic(ma.divide, self, other)

    def __rdiv__(self, other):
        if not isinstance(other, DataArray):
            res = self.__class__.new_from_obj(
                self, data=other / self._data, copy=True)
            if self._var is not None:
                res._var = self._var * other**2 / (self._data**2)**2
            return res
        # else:
        #     if other is a DataArray, it is already handled by __div__

    __radd__ = __add__
    __rmul__ = __mul__
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __add__.__doc__ = _arit_doc.format(name='addition', op='+')
    __sub__.__doc__ = _arit_doc.format(name='subtraction', op='-')
    __mul__.__doc__ = _arit_doc.format(name='multiplication', op='*')
    __div__.__doc__ = _arit_doc.format(name='division', op='/')
