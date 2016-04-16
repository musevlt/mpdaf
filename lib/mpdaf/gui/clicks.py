# -*- coding: utf-8 -*-

from __future__ import absolute_import
import logging
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


class ImageClicks(object):
    """Object used to save click on image plot."""

    def __init__(self, binding_id, filename=None):
        self._logger = logging.getLogger(__name__)
        self.filename = filename  # Name of the table fits file
        #                           where are saved the clicks values.
        self.binding_id = binding_id  # Connection id.
        self.p = []  # Nearest pixel of the cursor position along the y-axis.
        self.q = []  # Nearest pixel of the cursor position along the x-axis.
        self.x = []  # Corresponding nearest position along the x-axis
        #              (world coordinates)
        self.y = []  # Corresponding nearest position along the y-axis
        #              (world coordinates)
        self.data = []  # Corresponding image data value.
        self.id_lines = []  # Plot id (cross for cursor positions).

    def remove(self, ic, jc):
        """removes a cursor position"""
        d2 = (self.i - ic) * (self.i - ic) + (self.j - jc) * (self.j - jc)
        i = np.argmin(d2)
        line = self.id_lines[i]
        del plt.gca().lines[line]
        self.p.pop(i)
        self.q.pop(i)
        self.x.pop(i)
        self.y.pop(i)
        self.data.pop(i)
        self.id_lines.pop(i)
        for j in range(i, len(self.id_lines)):
            self.id_lines[j] -= 1
        plt.draw()

    def add(self, i, j, x, y, data):
        plt.plot(j, i, 'r+')
        self.p.append(i)
        self.q.append(j)
        self.x.append(x)
        self.y.append(y)
        self.data.append(data)
        self.id_lines.append(len(plt.gca().lines) - 1)

    def iprint(self, i):
        """prints a cursor positions"""
        self._logger.info('y=%g\tx=%g\tp=%d\tq=%d\tdata=%g', self.y[i],
                          self.x[i], self.p[i], self.q[i], self.data[i])

    def write_fits(self):
        """prints coordinates in fits table."""
        if self.filename != 'None':
            c1 = fits.Column(name='p', format='I', array=self.p)
            c2 = fits.Column(name='q', format='I', array=self.q)
            c3 = fits.Column(name='x', format='E', array=self.x)
            c4 = fits.Column(name='y', format='E', array=self.y)
            c5 = fits.Column(name='data', format='E', array=self.data)
            # tbhdu = fits.new_table(fits.ColDefs([c1, c2, c3, c4, c5]))
            coltab = fits.ColDefs([c1, c2, c3, c4, c5])
            tbhdu = fits.TableHDU(fits.FITS_rec.from_columns(coltab))
            tbhdu.writeto(self.filename, clobber=True, output_verify='fix')
            self._logger.info('printing coordinates in fits table %s',
                              self.filename)

    def clear(self):
        """disconnects and clears"""
        self._logger.info('disconnecting console coordinate printout...')
        plt.disconnect(self.binding_id)
        nlines = len(self.id_lines)
        for i in range(nlines):
            line = self.id_lines[nlines - i - 1]
            del plt.gca().lines[line]
        plt.draw()


class SpectrumClicks(object):
    """Object used to save click on spectrum plot."""

    def __init__(self, binding_id, filename=None):
        self.filename = filename  # Name of the table fits file where are
        # saved the clicks values.
        self.binding_id = binding_id  # Connection id.
        self.xc = []  # Cursor position in spectrum (world coordinates).
        self.yc = []  # Cursor position in spectrum (world coordinates).
        self.k = []  # Nearest pixel in spectrum.
        self.lbda = []  # Corresponding nearest position in spectrum
        # (world coordinates)
        self.data = []  # Corresponding spectrum data value.
        self.id_lines = []  # Plot id (cross for cursor positions).
        self._logger = logging.getLogger(__name__)

    def remove(self, xc):
        # removes a cursor position
        i = np.argmin(np.abs(self.xc - xc))
        line = self.id_lines[i]
        del plt.gca().lines[line]
        self.xc.pop(i)
        self.yc.pop(i)
        self.k.pop(i)
        self.lbda.pop(i)
        self.data.pop(i)
        self.id_lines.pop(i)
        for j in range(i, len(self.id_lines)):
            self.id_lines[j] -= 1
        plt.draw()

    def add(self, xc, yc, i, x, data):
        plt.plot(xc, yc, 'r+')
        self.xc.append(xc)
        self.yc.append(yc)
        self.k.append(i)
        self.lbda.append(x)
        self.data.append(data)
        self.id_lines.append(len(plt.gca().lines) - 1)

    def iprint(self, i):
        # prints a cursor positions
        msg = 'xc=%g\tyc=%g\tk=%d\tlbda=%g\tdata=%g' % (
            self.xc[i], self.yc[i], self.k[i], self.lbda[i], self.data[i])
        self._logger.info(msg)

    def write_fits(self):
        # prints coordinates in fits table.
        if self.filename != 'None':
            c1 = fits.Column(name='xc', format='E', array=self.xc)
            c2 = fits.Column(name='yc', format='E', array=self.yc)
            c3 = fits.Column(name='k', format='I', array=self.k)
            c4 = fits.Column(name='lbda', format='E', array=self.lbda)
            c5 = fits.Column(name='data', format='E', array=self.data)
            # tbhdu = fits.new_table(fits.ColDefs([c1, c2, c3, c4, c5]))
            coltab = fits.ColDefs([c1, c2, c3, c4, c5])
            tbhdu = fits.TableHDU(fits.FITS_rec.from_columns(coltab))
            tbhdu.writeto(self.filename, clobber=True)

            msg = 'printing coordinates in fits table %s' % self.filename
            self._logger.info(msg)

    def clear(self):
        # disconnects and clears
        msg = "disconnecting console coordinate printout..."
        self._logger.info(msg)

        plt.disconnect(self.binding_id)
        nlines = len(self.id_lines)
        for i in range(nlines):
            line = self.id_lines[nlines - i - 1]
            del plt.gca().lines[line]
        plt.draw()
