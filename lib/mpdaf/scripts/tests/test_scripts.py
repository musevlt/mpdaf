# -*- coding: utf-8 -*-

import os
from ..make_white_image import main
from ...tests.utils import get_data_file


def test_make_white_image(tmpdir, monkeypatch):
    cube = get_data_file('sdetect', 'minicube.fits')
    out = str(tmpdir.join('image.fits'))

    monkeypatch.setattr('sys.argv', ['make_white_image', '-v', cube, out])
    main()

    assert os.path.isfile(out)
