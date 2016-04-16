# -*- coding: utf-8 -*-

from __future__ import print_function

from __future__ import absolute_import
import argparse
from mpdaf.obj import Cube
from time import time


def make_white_image(inputfile, outputfile, verbose=False):
    t0 = time()
    print('Creating white light image {}'.format(outputfile))
    cube = Cube(inputfile)
    if verbose:
        cube.info()

    im = cube.mean(axis=0)
    im.write(outputfile)
    if verbose:
        print('Execution time {:.3f} seconds.'.format(time() - t0))


def main():
    parser = argparse.ArgumentParser(
        description='Make a white-light image from a cube, by computing the '
                    'mean for each pixel along the spectral axis.')
    parser.add_argument('input_cube', help='Input cube (FITS file).')
    parser.add_argument('output_image', help='Output image (FITS file).')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='verbose flag')
    args = parser.parse_args()
    make_white_image(args.input_cube, args.output_image, verbose=args.verbose)


if __name__ == '__main__':
    main()
