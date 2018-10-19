"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>

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

import astropy.units as u
import numpy as np
import pickle
import pytest
import warnings

from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from numpy import ma
from mpdaf.obj import DataArray, WaveCoord, WCS, Cube, Image, Spectrum
from numpy.testing import assert_array_equal, assert_allclose

from ...tests.utils import (generate_image, generate_cube, generate_spectrum,
                            assert_masked_allclose, get_data_file)


def test_fits_img():
    """DataArray class: Testing FITS image reading"""
    testimg = get_data_file('obj', 'a370II.fits')
    data = DataArray(filename=testimg)
    assert data.shape == (1797, 1909)
    assert data.ndim == 2
    assert data.dtype is None
    assert not data._loaded_data
    # Load data by accessing the .data attribute
    assert data.data.ndim == 2
    assert data.dtype.type == np.int16

    # Force dtype to float
    data = DataArray(filename=testimg, dtype=float)
    assert data.dtype == float
    assert data.data.dtype == float

    # Check that it can be read as Image but not with other classes
    assert Image(testimg).ndim == 2
    with pytest.raises(ValueError):
        Cube(testimg)
    with pytest.raises(ValueError):
        Spectrum(testimg)


def test_fits_spectrum():
    """DataArray class: Testing FITS spectrum reading"""
    testspe = get_data_file('obj', 'Spectrum_Variance.fits')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = DataArray(filename=testspe, ext=(0, 1),
                         fits_kwargs={'checksum': True})
    assert len([ww for ww in w if ww.category is AstropyUserWarning]) == 4
    assert data.shape == (4096,)
    assert data.ndim == 1

    # Load data by accessing the .data attribute
    assert data.data.ndim == 1
    assert data.dtype.type == np.float64
    assert data.var.dtype == np.float64

    # Check that it can be read as Spectrum but not with other classes
    assert Spectrum(testspe, ext=0).ndim == 1
    with pytest.raises(ValueError):
        Cube(testspe, ext=0)
    with pytest.raises(ValueError):
        Image(testspe, ext=0)


def test_float_conversion():
    # check that int are not converted to float
    testimg = get_data_file('obj', 'a370II.fits')
    data = DataArray(filename=testimg)
    assert str(data.data.dtype) == '>i2'

    # by default float32 is converted to float64
    testspe = get_data_file('obj', 'Spectrum_Variance.fits')
    data = DataArray(filename=testspe, ext=(0, 1))
    assert data.data.dtype == np.float64
    assert data.var.dtype == np.float64

    # avoid float64 conversion
    data = DataArray(filename=testspe, ext=(0, 1), convert_float64=False)
    assert str(data.data.dtype) == '>f4'
    assert str(data.var.dtype) == '>f4'


def test_invalid_file():
    """DataArray class: Testing invalid file reading"""
    with pytest.raises(IOError) as e:
        DataArray(filename='missing/file.test')
        assert e.exception.message == 'Invalid file: missing/file.test'


def test_from_hdulist():
    testimg = get_data_file('obj', 'a370II.fits')
    with fits.open(testimg) as hdul:
        data = DataArray(hdulist=hdul)
        assert data.shape == (1797, 1909)
        assert data.ndim == 2
        assert data.filename == testimg
        assert_allclose(data.data, hdul[0].data)


def test_from_ndarray():
    """DataArray class: Testing initialization from a numpy.ndarray"""

    nz = 2
    ny = 3
    nx = 4
    ntotal = nz * ny * nx
    data = np.arange(ntotal).reshape(nz, ny, nx)
    var = np.random.rand(ntotal).reshape(nz, ny, nx) - 0.5
    mask = var < 0.0

    # Create a DataArray with the above contents.
    d = DataArray(data=data, var=var, mask=mask)

    # Is the shape of the DataArray correct?
    assert d.shape == data.shape

    # Check that the enclosed data and variance arrays match
    # the arrays that were passed to the constructor.
    assert_allclose(d.data, data)
    assert_allclose(d.var, var)

    # Make sure that the enclosed data and var arrays are
    # masked arrays that both have the mask that was passed to the
    # constructor, and that these and the masked property are in
    # fact all references to the same mask.
    assert_array_equal(d.data.mask, mask)
    assert_array_equal(d.var.mask, mask)
    assert d.data.mask is d.mask and d.var.mask is d.mask

    # Test data with different dtypes
    d = DataArray(data=data.astype(np.float32), var=var.astype(np.float32),
                  mask=mask)
    assert d.dtype.type == np.float64
    assert d.var.dtype == np.float64

    d = DataArray(data=data.astype('>f4'), var=var.astype('>f4'), mask=mask)
    assert d.dtype.type == np.float64
    assert d.var.dtype == np.float64

    d = DataArray(data=data.astype(np.int32), var=var.astype(np.int32),
                  mask=mask)
    assert d.dtype.type == np.int32
    assert d.var.dtype == np.float64


def test_from_obj():
    """DataArray class: Testing initialization from an object"""
    d = DataArray(data=np.arange(10), var=np.ones(10))
    c = Cube.new_from_obj(d)
    assert c.shape == d.shape
    assert np.may_share_memory(c.data, d.data)
    assert_array_equal(c.data, d.data)
    assert_array_equal(c.var, d.var)

    data = np.zeros(10)
    c = Cube.new_from_obj(d, data=data, copy=True)
    assert not np.may_share_memory(c.data, data)
    assert_array_equal(c.data, data)
    assert_array_equal(c.var, d.var)


def test_copy():
    """DataArray class: Testing the copy method"""
    wcs = WCS(deg=True)
    wave = WaveCoord(cunit=u.angstrom)
    nz = 5
    ny = 4
    nx = 3
    n = nz * ny * nx
    data = np.arange(n).reshape(nz, ny, nx)
    var = np.arange(n).reshape(nz, ny, nx) / 10.0
    mask = data.astype(int) % 10 == 0
    cube1 = DataArray(data=data, var=var, mask=mask, wave=wave, wcs=wcs)
    cube2 = cube1.copy()

    assert_masked_allclose(cube1.data, cube2.data)
    assert_masked_allclose(cube1.var, cube2.var)
    assert_array_equal(cube1.data.mask, cube2.data.mask)
    assert_array_equal(cube1.var.mask, cube2.var.mask)

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert cube2.data.mask is cube2.mask
    assert cube2.var.mask is cube2.mask


def test_clone(cube):
    """DataArray class: Testing the clone method"""
    cube2 = cube.clone()
    assert cube.wcs.isEqual(cube2.wcs)
    assert cube.wave.isEqual(cube2.wave)
    assert cube2._data is None
    assert cube2._var is None
    assert cube2._mask is None


def test_clone_fits(minicube):
    """DataArray class: Testing the clone method with a FITS file"""
    im = minicube[0].clone()
    assert im.ndim == 2
    assert im.data_header['NAXIS'] == 2
    assert im.shape == minicube.shape[1:]
    assert 'NAXIS3' not in im.data_header

    sp = minicube[:, 20, 20]
    assert_array_equal(sp.abs().data, np.abs(sp._data))


def test_clone_with_data():
    """DataArray class: Testing the clone method with data"""

    # Define a couple of functions to be used to fill the data and
    # variance arrays of the cloned cube.

    def data_fn(shape, dtype=float):
        n = np.asarray(shape, dtype=int).prod()
        data = np.arange(n, dtype=dtype).reshape(shape) * 0.25
        mask = np.arange(n, dtype=int).reshape(shape) % 5 == 0
        return ma.array(data, mask=mask)

    def var_fn(shape, dtype=float):
        n = np.asarray(shape, dtype=int).prod()
        var = np.arange(n, dtype=dtype).reshape(shape) * 0.5
        mask = np.arange(n, dtype=int).reshape(shape) % 2 == 0
        return ma.array(var, mask=mask)

    # Create a generic cube of a specified shape.
    shape = (6, 3, 2)
    cube1 = generate_cube(shape=shape)

    # Create a shallow copy of the cube and fill its data and
    # variance arrays using the functions defined above.
    cube2 = cube1.clone(data_init=data_fn, var_init=var_fn)

    # The shared mask should be the union of the data and variance masks.
    expected_mask = np.logical_or(data_fn(shape).mask, var_fn(shape).mask)

    # Check that the data and variance arrays ended up containing
    # the expected ramps of values and the expected mask.
    assert_masked_allclose(cube2.data,
                           ma.array(data_fn(shape), mask=expected_mask))
    assert_masked_allclose(cube2.var,
                           ma.array(var_fn(shape), mask=expected_mask))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert cube2.data.mask is cube2.mask
    assert cube2.var.mask is cube2.mask


def test_pickle(cube, minicube):
    cube3 = Cube(get_data_file('obj', 'CUBE.fits'))
    del cube3.data_header['WCSAXES']
    for cube_ref in (cube, minicube, cube3):
        for obj in (cube_ref, cube_ref[1], cube_ref[:, 2, 2]):
            c = pickle.loads(pickle.dumps(obj))

            if obj.wcs is None:
                assert c.wcs is None
            else:
                assert obj.wcs.isEqual(c.wcs)

            if obj.wave is None:
                assert c.wave is None
            else:
                assert obj.wave.isEqual(c.wave)

            assert_array_equal(obj.data, c.data)
            assert_array_equal(obj.var, c.var)

            assert (list(obj.primary_header.items()) ==
                    list(c.primary_header.items()))


def test_set_var():
    """DataArray class: Testing the variance setter"""

    nz = 3
    ny = 4
    nx = 5
    ntotal = nz * ny * nx
    shape = (nz, ny, nx)
    data = np.arange(ntotal).reshape(nz, ny, nx)
    mask = data > ntotal // 2
    cube = generate_cube(data=data, var=None, mask=mask, shape=shape)

    # Create a variance array of the same shape as the cube.
    var = np.arange(ntotal, dtype=float).reshape(shape)

    # Assign the variance array to the cube and check that the
    # variance array in the cube matches it and that the assignment of
    # this unmasked numpy doesn't change the mask.
    cube.var = var
    assert_masked_allclose(cube.var, ma.array(var, mask=mask))

    # Assign the same variance array to the cube, but this time
    # make it a masked array with the original mask, and check that
    # both the variance and the mask arrays of the cube match it.
    cube.var = ma.array(var, mask=mask)
    assert_masked_allclose(cube.var, ma.array(var, mask=mask))

    # Also check that both the variance and data arrays have the
    # mask that was passed to generate_cube()
    assert_array_equal(cube.var.mask, mask)
    assert_array_equal(cube.data.mask, mask)

    # Remove the variance array and check that this worked.
    cube.var = None
    assert cube.var is None

    # Make sure that removing the variance array didn't affect
    # the mask of the data array.
    assert_array_equal(cube.data.mask, mask)

    # Check that the data and masked properties still both hold
    # references to the same mask array.
    assert cube.data.mask is cube.mask


def test_comparisons():
    """DataArray class: Testing comparison methods"""

    # Generate a test spectrum initialized with a linear ramp of values.
    n = 100
    data = np.arange(n, dtype=float)
    var = data / 10.0
    mask = data.astype(int) % 7 == 0
    spec = generate_spectrum(data=data, var=var, mask=mask)

    # [ __le__ ]

    s = spec <= n // 2

    # Verify that elements that had values <= n//2 are not masked,
    # (except those that were previously masked), and that values > n//2
    # are masked.
    expected_mask = np.logical_or(data > n // 2, mask)
    assert_masked_allclose(s.data, ma.array(data, mask=expected_mask))
    assert_masked_allclose(s.var, ma.array(var, mask=expected_mask))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask

    # [ __lt__ ]

    s = spec < n // 2

    # Verify that elements that had values < n//2 are not masked,
    # (except those that were previously masked), and that values >= n//2
    # are masked.
    expected_mask = np.logical_or(data >= n // 2, mask)
    assert_masked_allclose(s.data, ma.array(data, mask=expected_mask))
    assert_masked_allclose(s.var, ma.array(var, mask=expected_mask))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask

    # [ __ge__ ]

    s = spec >= n // 2

    # Verify that elements that had values >= n//2 are not masked,
    # (except those that were previously masked), and that values < n//2
    # are masked.
    expected_mask = np.logical_or(data < n // 2, mask)
    assert_masked_allclose(s.data, ma.array(data, mask=expected_mask))
    assert_masked_allclose(s.var, ma.array(var, mask=expected_mask))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask

    # [ __gt__ ]

    s = spec > n // 2

    # Verify that elements that had values > n//2 are not masked,
    # (except those that were previously masked), and that values <= n//2
    # are masked.
    expected_mask = np.logical_or(data <= n // 2, mask)
    assert_masked_allclose(s.data, ma.array(data, mask=expected_mask))
    assert_masked_allclose(s.var, ma.array(var, mask=expected_mask))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask


def test_getitem():
    """DataArray class: Testing the __getitem__ method"""

    nz = 50
    ny = 30
    nx = 20
    ntotal = nz * ny * nx

    data = np.arange(ntotal, dtype=float).reshape((nz, ny, nx))
    var = np.arange(ntotal, dtype=float).reshape((nz, ny, nx)) / 10.0
    mask = data.astype(int) % nx > 15

    cube = generate_cube(data=data, var=var, mask=mask,
                         wcs=WCS(deg=True), wave=WaveCoord(cunit=u.angstrom))

    # Extract a pixel from the cube and check that it has the expected value
    # from its position on the 1D ramp that was used to initialize the cube.
    za = 34
    ya = 5
    xa = 3
    pixel_a = xa + ya * nx + za * (nx * ny)
    s = cube[za, ya, xa]
    assert_allclose(s, pixel_a)

    # Extract a spectrum from the cube, and check that it has the expected
    # values.
    dz = 10
    zb = za + dz
    s = cube[za:zb, ya, xa]
    expected_spec = np.arange(pixel_a, pixel_a + dz * (nx * ny), nx * ny)

    assert_masked_allclose(s.data, ma.array(expected_spec, mask=False))
    assert_masked_allclose(s.var, ma.array(expected_spec / 10.0, mask=False))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask

    # Check that the wavelength of the first spectrum pixel matches that
    # of pixel za,ya,xa in the original cube.
    expected_wave = cube.wave.coord(za)
    actual_wave = s.wave.coord(0)
    assert_allclose(actual_wave, expected_wave)

    # Extract an image from the cube, and check that it has the
    # expected values.
    dz = 0
    dy = 3
    dx = 5
    zb = za + dz
    yb = ya + dy
    xb = xa + dx
    s = cube[za, ya:yb, xa:xb]
    first_row = np.arange(pixel_a, pixel_a + dx, dtype=float)
    row_offsets = np.arange(0, dy * nx, nx, dtype=float)
    expected_image = np.resize(first_row, (dy, dx)) + \
        np.repeat(row_offsets, dx).reshape((dy, dx))
    assert_masked_allclose(s.data, ma.array(expected_image, mask=False))
    assert_masked_allclose(s.var, ma.array(expected_image / 10.0, mask=False))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask

    # Check that the world coordinates of the first pixel of the
    # image match those of pixel(za,ya,xa) of the original cube.
    expected_sky = cube.wcs.pix2sky((ya, xa))
    actual_sky = s.wcs.pix2sky((0, 0))
    assert_allclose(actual_sky, expected_sky)

    # Extract a sub-cube from the cube, and check that it has the
    # expected values.
    dz = 2
    dy = 3
    dx = 5
    zb = za + dz
    yb = ya + dy
    xb = xa + dx
    s = cube[za:zb, ya:yb, xa:xb]
    spec_offsets = np.arange(0, dz * ny * nx, ny * nx, dtype=float)
    expected_cube = np.resize(expected_image, (dz, dy, dx)) + \
        np.repeat(spec_offsets, dy * dx).reshape((dz, dy, dx))
    assert_masked_allclose(s.data, ma.array(expected_cube, mask=False))
    assert_masked_allclose(s.var, ma.array(expected_cube / 10.0, mask=False))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask

    # Check that the world coordinates of the first pixel of the
    # image match those of pixel(za,ya,xa) of the original cube.
    expected_sky = cube.wcs.pix2sky((ya, xa))
    actual_sky = s.wcs.pix2sky((0, 0))
    assert_allclose(actual_sky, expected_sky)

    # Check that the wavelength of the first spectrum pixel matches that
    # of pixel za,ya,xa in the original cube.
    expected_wave = cube.wave.coord(za)
    actual_wave = s.wave.coord(0)
    assert_allclose(actual_wave, expected_wave)

    # Extract a sub-cube of values from the masked area of the cube
    # and check that all its data and variances are masked.
    s = cube[:, :, 16:]
    expected_mask = np.ones((nz, ny, nx - 16)) > 0
    assert_array_equal(s.data.mask, expected_mask)
    assert_array_equal(s.var.mask, expected_mask)

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask

    # Extract a sub-cube of values from the unmasked area of the cube
    # and check that all its data and variances are not masked.
    s = cube[:, :, :16]
    expected_mask = np.ones((nz, ny, 16)) < 1
    assert_array_equal(s.data.mask, expected_mask)
    assert_array_equal(s.var.mask, expected_mask)

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask


def test_setitem():
    """DataArray class: Testing the __setitem__ method"""

    nz = 50
    ny = 30
    nx = 20
    ntotal = nz * ny * nx

    data = np.arange(ntotal, dtype=float).reshape((nz, ny, nx))
    var = np.arange(ntotal, dtype=float).reshape((nz, ny, nx)) / 10.0
    mask = data.astype(int) % 3 == 1

    cube1 = generate_cube(data=data, var=var, mask=mask,
                          wcs=WCS(deg=True), wave=WaveCoord(cunit=u.angstrom))

    # ---------------------------------
    # Test the assignment of sub-cubes.

    cnz = 5
    cny = 3
    cnx = 2
    cn = cnz * cny * cnx
    slices = (slice(0, cnz), slice(0, cny), slice(0, cnx))

    # Start by assigning a 3D unmasked numpy array.
    cube2 = cube1.copy()
    d = (np.arange(cn) / 10.0).reshape(cnz, cny, cnx)
    cube2[slices] = d

    # Were the data assigned correctly without changing the mask?
    # LP: assigning data must change the mask (it is the case for the masked
    # array class)
    #     assert_masked_allclose(cube2.data[slices],
    #                            ma.array(d,mask=mask[slices]))
    assert_allclose(cube2._data[slices], d)

    # The corresponding variances aren't expected to change when a
    # simple data-array is assigned.
    assert_allclose(cube2._var[slices], cube1._var[slices])

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert cube2.data.mask is cube2.mask
    assert cube2.var.mask is cube2.mask

    # Next assign the same array, but as part of a 3D DataArray
    # with variances and a mask.
    v = d / 10.0
    m = d.astype(int) % 10 == 0
    cube2 = cube1.copy()
    c = generate_cube(data=d, var=v, mask=m, wcs=cube2[slices].wcs,
                      wave=cube2[slices].wave)
    cube2[slices] = c

    # Were the data, variances and the mask assigned correctly?
    assert_masked_allclose(cube2.data[slices], ma.array(d, mask=m))
    assert_masked_allclose(cube2.var[slices], ma.array(v, mask=m))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert cube2.data.mask is cube2.mask
    assert cube2.var.mask is cube2.mask

    # ----------------------------------
    # Test the assignment of sub-images.
    cny = 3
    cnx = 2
    cn = cny * cnx
    slices = (slice(0, 1), slice(0, cny), slice(0, cnx))

    # Start by assigning a 2D unmasked numpy array.
    cube2 = cube1.copy()
    d = (np.arange(cn) / 10.0).reshape(cny, cnx)
    cube2[slices] = d

    # Were the data assigned correctly without changing the mask.
    #     assert_masked_allclose(cube2.data[slices],
    #                            ma.array(d[np.newaxis,:,:],
    #                                        mask=mask[slices]))
    assert_allclose(cube2._data[slices], d[np.newaxis, :, :])

    # The corresponding variances aren't expected to change when a
    # simple data-array is assigned.
    assert_allclose(cube2._var[slices], cube1._var[slices])

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert cube2.data.mask is cube2.mask
    assert cube2.var.mask is cube2.mask

    # Next assign the same array, but as part of a 2D DataArray
    # with variances and a mask.
    v = d / 10.0
    m = d.astype(int) % 10 == 0
    cube2 = cube1.copy()
    c = generate_image(data=d, var=v, mask=m, wcs=cube2[slices].wcs)
    cube2[slices] = c

    # Were the data, variances and the mask assigned correctly?
    assert_masked_allclose(cube2.data[slices],
                           ma.array(d[np.newaxis, :, :],
                                    mask=m[np.newaxis, :, :]))
    assert_masked_allclose(cube2.var[slices],
                           ma.array(v[np.newaxis, :, :],
                                    mask=m[np.newaxis, :, :]))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert cube2.data.mask is cube2.mask
    assert cube2.var.mask is cube2.mask

    # -----------------------------------
    # Test the assignment of sub-spectra.

    cnz = 5
    cn = cnz
    slices = (slice(0, cnz), slice(0, 1), slice(0, 1))

    # Start by assigning a 2D unmasked numpy array.
    cube2 = cube1.copy()
    d = (np.arange(cnz) / 10.0).reshape(cnz, 1, 1)
    cube2[slices] = d

    # Were the data assigned correctly, and did the assignment clear
    # the mask of the corresponding elements?
    assert_masked_allclose(cube2.data[slices], ma.array(d, mask=False))

    # The corresponding variances aren't expected to change when a
    # simple data-array is assigned, but their mask should have been cleared.
    assert_masked_allclose(cube2.var[slices],
                           ma.array(cube1.var[slices].data, mask=False))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert cube2.data.mask is cube2.mask
    assert cube2.var.mask is cube2.mask


def test_get_wcs_header():
    """DataArray class: Testing the get_wcs_header method"""

    wcs = WCS(crpix=(15.0, 10.0), crval=(12.0, 14.0), cdelt=(0.1, 0.2),
              deg=True)
    im = generate_image(wcs=wcs)
    hdr = im.get_wcs_header()
    hdr_wcs = WCS(hdr)
    assert wcs.isEqual(hdr_wcs)


def test_get_data_hdu():
    """DataArray class: Testing the get_data_hdu method"""

    wcs = WCS(crpix=(15.0, 10.0), crval=(12.0, 14.0), cdelt=(0.1, 0.2),
              deg=True)
    wave = WaveCoord(crpix=2.0, cdelt=1.5, crval=12.5)
    cube = generate_cube(shape=(5, 3, 2), wcs=wcs, wave=wave)

    hdu = cube.get_data_hdu()
    hdr = hdu.header
    hdr_wcs = WCS(hdr)

    # Check that the WCS information taken from the header matches the
    # WCS information stored with the cube.
    assert cube.wcs.isEqual(hdr_wcs)

    # Check that the wavelength information taken from the header
    # matches the wavelength information stored with the cube.
    hdr_wave = WaveCoord(hdr)
    assert cube.wave.isEqual(hdr_wave)

    # Test the dtype conversion
    assert hdu.data.dtype == np.float32
    hdu = cube.get_data_hdu(convert_float32=False)
    assert hdu.data.dtype == np.float64


def test_get_stat_hdu():
    """DataArray class: Testing the get_stat_hdu method"""

    nz = 5
    ny = 20
    nx = 10

    # Create a cube, with a ramp of values assigned to the variance array,
    # and with some pixels masked.
    var = np.arange(nx * ny * nz, dtype=float).reshape((nz, ny, nx))
    mask = var.astype(int) % 7 == 0

    cube = generate_cube(data=0.1, var=var, mask=mask, wcs=WCS(deg=True),
                         wave=WaveCoord(cunit=u.angstrom))

    hdu = cube.get_stat_hdu()
    hdu_var = np.asarray(hdu.data, dtype=cube.dtype)
    assert_masked_allclose(ma.array(hdu_var, mask=mask), cube.var)

    # Test the dtype conversion
    assert hdu.data.dtype == np.float32
    hdu = cube.get_stat_hdu(convert_float32=False)
    assert hdu.data.dtype == np.float64


def test_write(tmpdir):
    """DataArray class: Testing the write method"""

    shape = (5, 4, 3)
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    var = data / 10.0
    mask = data.astype(int) % 10 == 0
    cube = generate_cube(data=data, var=var, mask=mask, wcs=WCS(deg=True),
                         wave=WaveCoord(cunit=u.angstrom))

    testfile = str(tmpdir.join('cube.fits'))
    cube.write(testfile, savemask='dq', checksum=True)

    # Verify that the contents of the file match the original cube.
    cube2 = Cube(testfile, fits_kwargs={'checksum': True})
    assert_masked_allclose(cube2.data, cube.data)
    assert_masked_allclose(cube2.var, cube.var)
    assert cube2.wcs.isEqual(cube.wcs)
    assert cube2.wave.isEqual(cube.wave)

    for k in ('DATASUM', 'CHECKSUM'):
        for hdr in (cube2.primary_header, cube2.data_header):
            assert k in hdr

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert cube2.data.mask is cube2.mask
    assert cube2.var.mask is cube2.mask

    # Test the dtype conversion and savemask=nan
    assert cube.dtype == np.float64
    assert cube2.dtype == np.float64
    assert fits.getval(testfile, 'BITPIX', ext=1) == -32
    assert fits.getval(testfile, 'BITPIX', ext=2) == -32

    cube.write(testfile, savemask='nan', checksum=True, convert_float32=False)
    cube3 = Cube(testfile, fits_kwargs={'checksum': True})
    assert str(cube3.data.dtype) == '>f8'
    assert fits.getval(testfile, 'BITPIX', ext=1) == -64
    assert fits.getval(testfile, 'BITPIX', ext=2) == -64
    assert_array_equal(cube3.mask, cube2.mask)


def test_sqrt():
    """DataArray class: Testing the sqrt method"""

    # Create a spectrum containing gaussian noise, offset by a mean
    # level that prevents it going negative. The lack of negatives
    # means that we can use the variance of its square rooted pixels
    # to estimate the expected variances of the square rooted data
    # values.

    mean = 100.0  # Mean of spectrum pixels.
    sdev = 2.0    # Standard deviation of pixel values around the mean.
    n = 100000    # The number of spectral pixels.
    data = np.random.normal(loc=mean, scale=sdev, size=n)
    masked_pixel = 12
    mask = np.isclose(np.arange(n), masked_pixel)  # Mask just one pixel.
    spec = generate_spectrum(data=data, mask=mask, var=sdev**2)

    # Get the square root of the spectrum.
    s = spec.sqrt()

    # Determine the mean and variance of the square rooted data.
    s_mean = s.data.mean()
    s_var = s.data.var()

    # Check that the mean is as close as expected to the theoretical
    # mean, noting that the standard deviation of the mean of the
    # square rooted data should be sqrt(0.25*sdev**2/mean/n) [for
    # for sdev=2,mean=100 and n=100000, this gives 0.0003.
    assert_allclose(s_mean, np.sqrt(mean), rtol=0.001)

    # Check that the mask didn't get changed.
    assert_array_equal(s.data.mask, mask)
    assert_array_equal(s.var.mask, mask)

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask

    # Given a sample of value x, picked from a distribution of variance vx,
    # compute the expected variance, vs, of sqrt(x).
    #
    #  vs = (d(sqrt(x))/dx)**2 * vx
    #     = (0.5/sqrt(x))**2 * vx
    #     = 0.25/x * vx.
    #
    # Sanity check this by seeing if the variance of square root of
    # the random pixels matches it.
    expected_var = 0.25 * sdev**2 / mean
    assert_allclose(s_var, expected_var, rtol=0.05)

    # Check that the recorded variances match the expected variance.
    assert_allclose(np.ones(n) * s_var, expected_var, rtol=0.1)

    # Generate another spectrum, but this time fill it with positive
    # and negative values.
    n = 10        # The number of spectral pixels.
    sdev = 0.5    # The variance to be recorded with each pixel.
    data = np.hstack((np.ones(n // 2) * -1.0, np.ones(n // 2) * 1.0))
    spec = generate_spectrum(data=data, var=sdev**2)

    s = spec.sqrt()

    # Check that the square-root masked the values that were negative
    # and not those that weren't.
    expected_mask = data < 0.0
    assert_array_equal(s.data.mask, expected_mask)
    assert_array_equal(s.var.mask, expected_mask)

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert s.data.mask is s.mask and s.var.mask is s.mask


def test_abs():
    """DataArray class: Testing the abs method"""

    # Create a spectrum containing a ramp that goes from negative to
    # positive integral values, stored as floats. Assign a constant
    # variance to all points, and mask a couple of points either side
    # of zero.
    ramp = np.arange(10.0, dtype=float) - 5
    var = 0.5
    mask = np.logical_and(ramp > -2, ramp < 2)
    spec1 = generate_spectrum(ramp, var=var, mask=mask)
    spec2 = spec1.abs()

    # Check that the unmasked values of the data array have been
    # replaced by their absolute values, that the variance array
    # hasn't been changed, and that the shared mask has not been
    # unchanged.
    assert_masked_allclose(spec2.data, abs(spec1.data))
    assert_masked_allclose(spec2.var, spec1.var)

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert spec2.data.mask is spec2.mask
    assert spec2.var.mask is spec2.mask


def test_unmask():
    """DataArray class: Testing the unmask method"""

    # Create a spectrum containing a simple ramp from negative to
    # positive values, with negative values masked, and one each of
    # the infinity and nan special values.
    n = 10
    data = np.arange(n, dtype=float) - 5
    var = data / 10.0
    mask = data < 0
    data[0] = np.inf
    data[1] = np.nan
    spec = generate_spectrum(data, var=var, mask=mask)

    # Clear the mask, keeping just the inf and nan values masked.
    spec.unmask()

    # Check that only the Inf and Nan values are still masked
    # and that neither the data nor the variance values have been
    # changed.
    expected_mask = np.arange(n, dtype=int) < 2
    assert_masked_allclose(spec.data, ma.array(data, mask=expected_mask))
    assert_masked_allclose(spec.var, ma.array(var, mask=expected_mask))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask


def test_mask_variance():
    """DataArray class: Testing the mask_variance method"""

    # Create a spectrum whose variance array contains a positive ramp
    # of integral values stored as floats, and with a data array where
    # values below a specified threshold, chosen to be half way between
    # data values, are masked.
    n = 10
    data = np.arange(n, dtype=float)
    var = np.arange(n, dtype=float)
    lower_lim = 1.5
    spec = generate_spectrum(data=data, var=var, mask=var < lower_lim)

    # Mask all variances above a second threshold that is again chosen
    # to be half way between two known integral values in the array.
    upper_lim = 6.5
    spec.mask_variance(upper_lim)

    # Check that the originally masked values are still masked, and that
    # the values above the upper threshold are now also masked, and that
    # values in between them are not masked.
    expected_mask = np.logical_or(var < lower_lim, var > upper_lim)
    assert_masked_allclose(spec.data, ma.array(data, mask=expected_mask))
    assert_masked_allclose(spec.var, ma.array(var, mask=expected_mask))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask


def test_mask_selection():
    """DataArray class: Testing the mask_selection method"""

    # Create a test spectrum with a ramp for the data array
    # and all values below a specified limit masked.
    n = 10
    data = np.arange(n, dtype=float)
    var = data / 10.0
    lower_lim = 1.5
    spec = generate_spectrum(data=data, var=var, mask=data < lower_lim)

    # Apply the mask_selection method to mask values above a different
    # threshold.
    upper_lim = 6.5
    spec.mask_selection(np.where(data > upper_lim))

    # Check that the resulting data mask is the union of the
    # originally mask and the newly selected mask.
    expected_mask = np.logical_or(data < lower_lim, data > upper_lim)
    assert_masked_allclose(spec.data, ma.array(data, mask=expected_mask))
    assert_masked_allclose(spec.var, ma.array(var, mask=expected_mask))

    # Check that the data, var and masked properties all hold
    # references to the same mask array.
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask


def test_shared_masks():
    """DataArray class: Testing shared masks"""

    n = 50
    old_data = np.arange(n, dtype=float)
    old_var = np.arange(n, dtype=float) * 0.5 - n / 4.0
    old_mask = old_var < 0.0

    # Add infinite values to currently masked elements of the data
    # and var arrays and create a mask that just masks these values.
    old_data[0] = np.Inf
    old_var[1] = np.Inf

    # Create a spectrum DataArray with the above contents.
    template_spec = DataArray(data=old_data, var=old_var, mask=old_mask)

    # ----------------------------------------------------------------
    # Assign a numpy.ndarray of the same shape to DataArray.var:
    #
    # Create a simple numpy array with values that are chosen to be
    # distinct from the current variances. Also add an Inf and a Nan
    # at indexes that are not masked in the original spectrum.
    new_var = np.arange(n) * 0.01
    new_var[30] = np.inf
    new_var[31] = np.nan

    # What should the shared mask of the spectrum be after we assign
    # the above array to the .var property? When a simple ndarray is
    # assigned to the var property, the mask should remain the same,
    # except where the new variance array contains extra Inf or Nan
    # values.
    expected_mask = old_mask.copy()
    #    expected_mask[30:32] = True
    # In the end, we decided to create the mask only from the data.
    # But the user can set the var attribute with a masked array,
    # In this case, the given mask will be taken into account.

    # Assign the new array to the var property of the spectrum, then
    # check the resulting data, var and masked properties.
    spec = template_spec.copy()
    spec.var = new_var
    assert_masked_allclose(spec.var, ma.array(new_var, mask=expected_mask))
    assert_masked_allclose(spec.data, ma.array(old_data, mask=expected_mask))
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask

    # ----------------------------------------------------------------
    # Assign a MaskedArray of the same shape to DataArray.var:
    #
    # Create a masked array with values that are chosen to be distinct
    # from the current variances, and a mask that doesn't flag any
    # values. Then add an Inf and a Nan at indexes that are not masked
    # in the original spectrum, along with a single masked value.
    new_var = ma.array(np.arange(n) * 0.01, mask=False, copy=True)
    new_var[30] = np.inf
    new_var[31] = np.nan
    new_var[32] = ma.masked
    new_var = ma.masked_invalid(new_var, copy=False)

    # What should the shared mask of the spectrum be after we assign
    # the above array to the .var property? When a masked array of the
    # same shape is assigned to the var property, the resulting mask
    # should be the union of the original shared mask, the mask of the
    # new variance array
    expected_mask = old_mask.copy()
    expected_mask[30:33] = True
    expected_mask[32] = True

    # Assign the new array to the var property of the spectrum, then
    # check the resulting data, var and masked properties.
    spec = template_spec.copy()
    spec.var = new_var
    assert_masked_allclose(spec.var,
                           ma.array(new_var.data, mask=expected_mask))
    assert_masked_allclose(spec.data, ma.array(old_data, mask=expected_mask))
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask

    # ----------------------------------------------------------------
    # Assign a numpy.ndarray of the same shape to DataArray.data:
    #
    # Create a simple numpy array with values that are chosen to be
    # distinct from the current variances. Also add an Inf and a Nan
    # at indexes that are not masked in the original spectrum.
    new_data = np.arange(n) * 0.3
    new_data[30] = np.inf
    new_data[31] = np.nan

    # What should the shared mask of the spectrum be after we assign
    # the above array to the .data property? When a simple ndarray is
    # assigned to the data property, the mask should remain the same,
    # except where the new data array contains extra Inf or Nan
    # values.
    expected_mask = np.zeros(old_mask.shape, dtype=bool)
    expected_mask[30:32] = True

# NO the mask should not remain the same ...

    # Assign the new array to the data property of the spectrum, then
    # check the resulting data, var and masked properties.
    spec = template_spec.copy()
    spec.data = new_data
    assert_masked_allclose(spec.var, ma.array(old_var, mask=expected_mask))
    assert_masked_allclose(spec.data, ma.array(new_data, mask=expected_mask))
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask

    # ----------------------------------------------------------------
    # Assign a MaskedArray of the same shape to DataArray.data:
    #
    # Create a masked array with values that are chosen to be distinct
    # from the current spectrum data, and a mask that doesn't flag any
    # values. Then add an Inf and a Nan at indexes that are not masked
    # in the original spectrum, along with a single masked value.
    new_data = np.array(np.arange(n) * 0.3, copy=True)
    new_data[30] = np.inf
    new_data[31] = np.nan
    new_data[32] = ma.masked

    # What should the shared mask of the spectrum be after we assign
    # the above array to the .data property? When a masked array is
    # assigned to the data property, the resulting mask should be the
    # mask of the new array, along with additions for each element in
    # the new array

# The invalid pixel of the variance are not taken into account

    expected_mask = np.ones(spec.shape) < 1  # All False
    expected_mask[30:33] = True
    # expected_mask = np.logical_or(expected_mask, ~(np.isfinite(old_var)))

    # Assign the new array to the data property of the spectrum, then
    # check the resulting data, var and masked properties.

    spec = template_spec.copy()
    spec.data = new_data
    assert_masked_allclose(spec.var, ma.array(old_var, mask=expected_mask))
    assert_masked_allclose(spec.data,
                           ma.array(new_data, mask=expected_mask))
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask

    # ----------------------------------------------------------------
    # Assign a numpy.ndarray of a different shape to DataArray.var:
    #
    # The assignment of a variance array of a different size to the
    # data array is supposed to raise an exception, so just create
    # an array that has a different size.
    new_var = np.arange(n // 2)

    # Try to assign the new array to the var property of the spectrum
    # and make sure that this generates an exception, because the
    # shape of the variance array is incompatible with the shape of
    # the current data array.
    spec = template_spec.copy()
    try:
        spec.var = new_var
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched variance array shape not caught')

    # ----------------------------------------------------------------
    # Assign a MaskedArray.ndarray of a different shape to DataArray.var:
    #
    # The assignment of a variance array of a different size to the
    # data array is supposed to raise an exception, so just create
    # an array that has a different size.
    new_var = ma.array(np.arange(n // 2), mask=False)

    # Try to assign the new array to the var property of the spectrum
    # and make sure that this generates an exception, because the
    # shape of the variance array is incompatible with the shape of
    # the current data array.
    spec = template_spec.copy()
    try:
        spec.var = new_var
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched variance array shape not caught')

    # ----------------------------------------------------------------
    # Assign a numpy.ndarray of a different shape to DataArray.data:
    #
    # Create a simple numpy array with a different size and values
    # that are chosen to be distinct from the current variances. Also
    # add an Inf and a Nan at indexes that are not masked in the
    # original spectrum.
    new_n = n + 5
    new_data = np.arange(new_n) * 0.3
    new_data[30] = np.inf
    new_data[31] = np.nan

    # What should the shared mask of the spectrum be after we assign
    # the above array to the .data property? When a simple ndarray of
    # a different size is assigned to the data property, the resulting
    # mask should be clear except where the new data array contains
    # Inf or Nan.
    expected_mask = np.ones(new_n) < 1
    expected_mask[30:32] = True

    # Assign the new array to the data property of the spectrum, then
    # check the resulting data, var and masked properties. Note that
    # in this case the old variance array should remain in place
    # with the old mask.
    spec = template_spec.copy()

    try:
        spec.data = new_data
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched data array shape not caught')

    # Is it possible to assign data with an array of a different size ?

#     spec.data = new_data
#     assert_masked_allclose(spec.var, ma.array(old_var, mask=old_mask))
#     assert_masked_allclose(spec.data, ma.array(new_data, mask=expected_mask))
#     assert spec.data.mask is spec.mask
#     assert spec.var.mask is not spec.mask
#
#     #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
#     # Now that the size of the data array has been changed, try assigning
#     # a new variance array of the same size. Give this a couple of
#     # invalid values at different places than the newly installed data
#     # array.
#
#     new_var = np.arange(new_n) * 0.01
#     new_var[35] = np.inf
#     new_var[36] = np.nan
#
#     # What should the shared mask of the spectrum be after we assign
#     # the above array to the .var property? When a simple ndarray is
#     # assigned to the var property, the mask should remain the same,
#     # except where the new variance array contains extra Inf or Nan
#     # values.
#
#     expected_mask = spec.mask.copy()
#     expected_mask[35:37] = True
#
#     # Assign the new array to the var property of the spectrum, then
#     # check the resulting data, var and masked properties, noting
#     # that now the final variance array should use a reference to
#     # the shared mask array.
#
#     spec.var = new_var
#     assert_masked_allclose(spec.var, ma.array(new_var, mask=expected_mask))
#     assert_masked_allclose(spec.data, ma.array(new_data, mask=expected_mask))
#     assert_true(spec.data.mask is spec.mask and
#                 spec.var.mask is spec.mask)

    # ----------------------------------------------------------------
    # Assign a MaskedArray of a different shape to DataArray.data:
    #
    # Create a masked array of a different size, with values that are
    # chosen to be distinct from the current spectrum data, and a mask
    # that doesn't flag any values. Then add an Inf and a Nan at
    # indexes that are not masked in the original spectrum, along with
    # a single masked value.

#     new_n = n+5
#     new_data = ma.array(np.arange(new_n)*0.3, mask=False, copy=True)
#     new_data[30] = np.inf
#     new_data[31] = np.nan
#     new_data[32] = ma.masked
#
#     # What should the shared mask of the spectrum be after we assign
#     # the above array to the .data property? When a masked array of a
#     # different size is assigned to the data property, the resulting
#     # mask should be the mask of the new array, along with additions
#     # for each element in the new array that are Inf or Nan.
#
#     expected_mask = np.ones(new_n) < 1  # All False
#     expected_mask[30:33] = True
#
#     # Assign the new array to the data property of the spectrum, then
#     # check the resulting data, var and masked properties. Note that
#     # in this case the old variance array should remain in place with
#     # the original mask.
#
#     spec = template_spec.copy()
#     spec.data = new_data
#     assert_masked_allclose(spec.var, ma.array(old_var, mask=old_mask))
#     assert_masked_allclose(spec.data,
#                            ma.array(new_data.data, mask=expected_mask))
#     assert spec.data.mask is spec.mask
#     assert spec.var.mask is not spec.mask
#
#     # ----------------------------------------------------------------
#     # Directly modify the masks of the data, var and masked properties.
#     #
#     # When a new value is assigned to an element of the mask of the
#     # data, var or masked properties, the change should be visible in
#     # all of these masks, which must remain as references to a single
#     # array.
#
#     spec = template_spec.copy()
#
#     # Try changing the mask of the data property, and check that this
#     # changes the value of the masks of all of the properties.
#
#     toggled_value = not spec.data.mask[2]
#     expected_mask = spec.mask.copy()
#     expected_mask[2] = toggled_value
#     spec.data.mask[2] = toggled_value
#     assert_array_equal(spec.data.mask, expected_mask)
#     assert_array_equal(spec.var.mask, expected_mask)
#     assert_array_equal(spec.mask, expected_mask)
#     assert_true(spec.data.mask is spec.mask and
#                 spec.var.mask is spec.mask)
#
#     # Now check the effect of changing the mask of the var property.
#
#     toggled_value = not spec.var.mask[2]
#     expected_mask = spec.mask.copy()
#     expected_mask[2] = toggled_value
#     spec.var.mask[2] = toggled_value
#     assert_array_equal(spec.data.mask, expected_mask)
#     assert_array_equal(spec.var.mask, expected_mask)
#     assert_array_equal(spec.mask, expected_mask)
#     assert_true(spec.data.mask is spec.mask and
#                 spec.var.mask is spec.mask)
#
#     # Now check the effect of changing the masked property.
#
#     toggled_value = not spec.mask[2]
#     expected_mask = spec.mask.copy()
#     expected_mask[2] = toggled_value
#     spec.mask[2] = toggled_value
#     assert_array_equal(spec.data.mask, expected_mask)
#     assert_array_equal(spec.var.mask, expected_mask)
#     assert_array_equal(spec.mask, expected_mask)
#     assert_true(spec.data.mask is spec.mask and
#                 spec.var.mask is spec.mask)

    # ----------------------------------------------------------------
    # Assign a new mask of the same size to the masked property.
    spec = template_spec.copy()

    # Create a new mask array.
    new_mask = np.logical_not(spec.mask.copy())

    # After assigning the above array to the masked property,
    # the final mask should be this mask plus masked elements
    # for any invalid values in the data and var properties.

# if user assignes the mask attribute, this mask is simply used

    expected_mask = new_mask
#     expected_mask = np.logical_or(expected_mask,
#                                   ~(np.isfinite(spec.data.data)))
#     expected_mask = np.logical_or(expected_mask,
#                                   ~(np.isfinite(spec.var.data)))

    # Assign the new mask to the masked property and check that the
    # expected mask is install and that a reference to a single copy
    # of this is used by all properties

    spec.mask = new_mask
    assert_array_equal(spec.data.mask, expected_mask)
    assert_array_equal(spec.var.mask, expected_mask)
    assert_array_equal(spec.mask, expected_mask)
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask

    # ----------------------------------------------------------------
    # Attempt to assign a mask of a different size to the masked property.

    spec = template_spec.copy()

    # Create a new mask array.
    new_mask = ma.make_mask_none(n + 5)

    # It is illegal to assign a mask of a different size than the data
    # array, so check that an exception is raised if we try to do this.
    try:
        spec.mask = new_mask
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched mask array shape not caught')

    # ----------------------------------------------------------------
    # Try assigning np.masked to elements of the data and var
    # properties.

    spec = template_spec.copy()

    # We will use ma.masked to mask a couple of elements of
    # data. Compute the expected mask.
    expected_mask = spec.mask.copy()
    expected_mask[30:34] = True

    # Mask two elements of data and check that this masks those
    # elements without changing their values, and that the mask
    # continues to be shared with the var and masked properties.
    spec.data[30:32] = ma.masked
    spec.var[32:34] = ma.masked
    assert_masked_allclose(spec.data, ma.array(old_data, mask=expected_mask))
    assert_masked_allclose(spec.var, ma.array(old_var, mask=expected_mask))
    assert_allclose(spec.data.data[30:32], old_data[30:32])
    assert_allclose(spec.var.data[32:34], old_var[32:34])
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask

    # ----------------------------------------------------------------
    # Check that in-place arithmetic operations work when they mask
    # elements.

    spec = template_spec.copy()

    # We will multiply the data and variance arrays by values
    # that flag two elements each. Compute the expected mask.
    expected_mask = spec.mask.copy()
    expected_mask[30] = True
    expected_mask[32] = True

    # Compute the expected data and variance arrays.
    new_data = old_data.copy()
    new_data[30:32] *= 2.0
    new_var = old_var.copy()
    new_var[32:34] *= 2.0

    # Multiply two elements each of data and var by masked arrays that
    # have one element unmasked and the other masked, and check the
    # results.
    spec.data[30:32] *= ma.array([2.0, 2.0], mask=[True, False])
    spec.var[32:34] *= ma.array([2.0, 2.0], mask=[True, False])
    assert_masked_allclose(spec.data, ma.array(new_data, mask=expected_mask))
    assert_masked_allclose(spec.var, ma.array(new_var, mask=expected_mask))
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask


def test_non_masked_data():
    """DataArray class: Testing non-masked data"""

    # Create 1D data, variance and mask arrays with the following
    # dimensions.
    n = 50
    old_data = np.arange(n, dtype=float)
    old_var = np.arange(n, dtype=float) * 0.5 - n / 4.0

    # Add infinite values to currently masked elements of the data
    # and var arrays and create a mask that just masks these values.
    old_data[0] = np.Inf
    old_var[1] = np.Inf

    # Create a spectrum DataArray with the above contents and explicitly
    # ask for no mask to be created.
    template_spec = DataArray(data=old_data, var=old_var, mask=ma.nomask)

    # Check that all of the data, var and masked properties have no mask.
    assert_masked_allclose(template_spec.data,
                           ma.array(old_data, mask=ma.nomask))
    assert_masked_allclose(template_spec.var,
                           ma.array(old_var, mask=ma.nomask))
    assert template_spec.mask is ma.nomask

    # Assign a new ndarray data array and a new ndarray variance array
    # of the same size, and check that this doesn't trigger masks to be
    # reinstated.
    new_data = np.arange(n, dtype=float) * 0.3
    new_var = np.arange(n, dtype=float) * 0.1
    spec = template_spec.copy()
    spec.data = ma.MaskedArray(new_data, mask=np.ma.nomask)
    spec.var = new_var

    assert_masked_allclose(spec.data, ma.array(new_data, mask=ma.nomask))
    assert_masked_allclose(spec.var, ma.array(new_var, mask=ma.nomask))
    assert spec.mask is ma.nomask

    # Assign ndarray arrays of a new size to the data and var properties.

#     new_n = n + 5
#     new_data = np.arange(new_n, dtype=float) * 0.3
#     new_var =  np.arange(new_n, dtype=float) * 0.1
#     spec = template_spec.copy()
#     spec.data = new_data
#     spec.var = new_var
#
#     assert_masked_allclose(spec.data, ma.array(new_data, mask=ma.nomask))
#     assert_masked_allclose(spec.var, ma.array(new_var, mask=ma.nomask))
#     assert spec.mask is ma.nomask

    # Assign a masked array of the same size to the data property.
    new_mask = ma.make_mask_none(n)
    new_data = ma.array(np.arange(n) * 0.3, mask=new_mask)
    spec = template_spec.copy()
    spec.data = new_data

    # The mask of the new data array should trigger a switch to
    # masking. Compute the expected mask, which should be new_mask
    # with the addition of an element for the infinity in the original
    # variance array that is still installed.
    expected_mask = new_mask
    expected_mask[1] = True

    assert_masked_allclose(spec.data,
                           ma.array(new_data.data, mask=expected_mask))
    assert_masked_allclose(spec.var,
                           ma.array(old_var, mask=expected_mask))
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask

    # Assign a masked array of the same size to the var property.
    new_mask = ma.make_mask_none(n)
    new_var = ma.array(np.arange(n) * 0.1, mask=new_mask)
    spec = template_spec.copy()
    spec.var = new_var

    # The mask of the new variance array should trigger a switch to
    # masking. Compute the expected mask, which should be new_mask
    # with the addition of a flag for the infinity in the original
    # data array that is still installed.

    expected_mask = new_mask
    # expected_mask[0] = True

    assert_masked_allclose(spec.data,
                           ma.array(old_data, mask=expected_mask))
    assert_masked_allclose(spec.var,
                           ma.array(new_var.data, mask=expected_mask))
    assert spec.data.mask is spec.mask
    assert spec.var.mask is spec.mask


def test_float32_ndarray():

    nz = 2
    ny = 3
    nx = 4
    ntotal = nz * ny * nx
    data = np.arange(ntotal).reshape(nz, ny, nx).astype(np.float32)
    var = np.random.rand(ntotal).reshape(nz, ny, nx) - 0.5
    mask = var < 0.0

    # Create a DataArray with the above contents.
    d = DataArray(data=data, var=var, mask=mask)

    # Is the shape of the DataArray correct?
    assert d.shape == data.shape

    # Check that the enclosed data and variance arrays match
    # the arrays that were passed to the constructor.
    assert_allclose(d.data, data)
    assert_allclose(d.var, var)

    # Make sure that the enclosed data and var arrays are
    # masked arrays that both have the mask that was passed to the
    # constructor, and that these and the masked property are in
    # fact all references to the same mask.
    assert_array_equal(d.data.mask, mask)
    assert_array_equal(d.var.mask, mask)
    assert d.data.mask is d.mask and d.var.mask is d.mask


def test_replace_data():
    """Test replacement of the data array"""
    testimg = get_data_file('obj', 'a370II.fits')
    data = DataArray(filename=testimg)
    data.data = np.zeros(data.shape)
    assert_array_equal(data.data, 0)

    data = DataArray(filename=testimg)
    new = ma.zeros(data.shape)
    new[:5, :5] = ma.masked
    data.data = new

    assert_array_equal(data.data, 0)
    assert ma.count_masked(data.data) == 5 * 5
    assert np.all(data.mask[:5, :5])
