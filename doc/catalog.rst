**************
Catalog object
**************

`~mpdaf.sdetect.Catalog` is a subclass of Table class from astropy.table.

.. warning:: Catalog class is currently under development

A catalog object can be read from a file (by using the astropy.table constructor `~mpdaf.sdetect.Catalog.read`)
or it can be created from a list of `~mpdaf.sdetect.Source` objects
(by using `~mpdaf.sdetect.Catalog.from_sources` or `~mpdaf.sdetect.Catalog.from_path`).

In order to present the `~mpdaf.sdetect.Catalog` class, we will first run `~mpdaf.sdetect.muselet`, a detection code that will return you a list of sources::

  In [1]: from mpdaf.sdetect import muselet

  In [2]: cont, single, raw = muselet('../data/sdetect/minicube.fits',nbcube=False, del_sex=True)                       
  SExtractor version 2.19.5 (2014-03-21)
  [INFO] muselet - Opening: ../data/sdetect/minicube.fits
  [INFO] muselet - STEP 1: creates white light, variance, RGB and narrow-band images
  [INFO] muselet - STEP 2: runs SExtractor on white light, RGB and narrow-band images
  ...
  [INFO] muselet - STEP 3: merge SExtractor catalogs and measure redshifts
  [INFO] muselet - cleaning below inverse variance 0.00408195029013
  [INFO] muselet - 1 continuum lines detected
  [INFO] muselet - 8 single lines detected
  [INFO] muselet - estimating the best redshift
  [INFO] crack_z: z=0.085800 err_z=0.000085

Then, we create our catalog from the list of single lines.
The new catalog will contain all data stored in the primary headers
and in the tables extensions of the sources:

 - a column per header fits
 - two columns per magnitude band [BAND] [BAND]_ERR
 - three columns per redshiftZ_[Z_DESC], Z_[Z_DESC]_MIN and Z_[Z_DESC]_MAX
 - several columns per line.

The lines columns depend of the format.
   
By default the columns names are created around unique LINE name [LINE]_[LINES columns names].

But it is possible to use a working format.
[LINES columns names]_xxx where xxx is the number of lines present in each source.

See the differences between the two format on the single lines detected by muselet::

  In [3]: from mpdaf.sdetect import Catalog

  In [4]: cat = Catalog.from_sources(single, fmt='default')
  
  In [5]: print cat
   ID        RA        DEC      ORIGIN ORIGIN_V      CUBE      Z_EMI   Z_EMI_MAX ... [SII]_FLUX_ERR [SII]_LBDA_OBS [SII]_LBDA_OBS_ERR _FLUX _FLUX_ERR _LBDA_OBS _LBDA_OBS_ERR
  unitless    deg        deg                                    unitless  unitless ...                   Angstrom         Angstrom                       Angstrom    Angstrom  
  -------- ---------- ---------- ------- -------- ------------- -------- --------- ... -------------- -------------- ------------------ ----- --------- --------- -------------
         6 63.3561076 10.4661661 muselet      2.1 minicube.fits   0.0856    0.0857 ...           32.5        7289.89               1.25 279.3      25.4   7297.39          1.25
         4 63.3554029 10.4647022 muselet      2.1 minicube.fits   0.0860    0.0860 ...             --             --                 --    --        --        --            --
         1 63.3559249 10.4653691 muselet      2.1 minicube.fits   0.0860    0.0860 ...           55.8        7292.39               1.25 518.0      38.5   7307.39          1.25
         2 63.3556900 10.4646378 muselet      2.1 minicube.fits   0.0862    0.0862 ...           50.6        7293.64               1.25 454.9      35.7   6841.14          1.25
         3 63.3552677 10.4657851 muselet      2.1 minicube.fits   0.0864    0.0864 ...           46.8        7294.89               1.25 677.4      29.4   7311.14          1.25
         5 63.3567159 10.4656965 muselet      2.1 minicube.fits   0.0862    0.0863 ...           45.7        7293.64               1.25    --        --        --            --
         7 63.3559897 10.4646136 muselet      2.1 minicube.fits       --        -- ...             --             --                 -- 415.1      36.2   7122.39          1.25
         8 63.3549327 10.4656274 muselet      2.1 minicube.fits       --        -- ...             --             --                 --    --        --        --            --

  In [6]: cat.colnames
  Out[6]: 
  ['ID',
   'RA',
   'DEC',
   'ORIGIN',
   'ORIGIN_V',
   'CUBE',
   'Z_EMI',
   'Z_EMI_MAX',
   'Z_EMI_MIN',
   'Halpha_FLUX',
   'Halpha_FLUX_ERR',
   'Halpha_LBDA_OBS',
   'Halpha_LBDA_OBS_ERR',
   'Lya/[OII]_FLUX',
   'Lya/[OII]_FLUX_ERR',
   'Lya/[OII]_LBDA_OBS',
   'Lya/[OII]_LBDA_OBS_ERR',
   '[NII]_FLUX',
   '[NII]_FLUX_ERR',
   '[NII]_LBDA_OBS',
   '[NII]_LBDA_OBS_ERR',
   '[SII]2_FLUX',
   '[SII]2_FLUX_ERR',
   '[SII]2_LBDA_OBS',
   '[SII]2_LBDA_OBS_ERR',
   '[SII]_FLUX',
   '[SII]_FLUX_ERR',
   '[SII]_LBDA_OBS',
   '[SII]_LBDA_OBS_ERR',
   '_FLUX',
   '_FLUX_ERR',
   '_LBDA_OBS',
   '_LBDA_OBS_ERR'] 
  
  In [7]: cat = Catalog.from_sources(single, fmt='working')

  In [8]: print cat
     ID        RA        DEC      ORIGIN ORIGIN_V      CUBE      Z_EMI   Z_EMI_MAX ... LBDA_OBS005 LBDA_OBS_ERR005 LINE005  FLUX006 FLUX_ERR006 LBDA_OBS006 LBDA_OBS_ERR006 LINE006 
  unitless    deg        deg                                    unitless  unitless ...   Angstrom      Angstrom    unitless                       Angstrom      Angstrom    unitless
  -------- ---------- ---------- ------- -------- ------------- -------- --------- ... ----------- --------------- -------- ------- ----------- ----------- --------------- --------
         6 63.3561076 10.4661661 muselet      2.1 minicube.fits   0.0856    0.0857 ...          --              --               --          --          --              --         
         4 63.3554029 10.4647022 muselet      2.1 minicube.fits   0.0860    0.0860 ...          --              --               --          --          --              --         
         1 63.3559249 10.4653691 muselet      2.1 minicube.fits   0.0860    0.0860 ...     6839.89            1.25            442.5        29.6     7121.14            1.25         
         2 63.3556900 10.4646378 muselet      2.1 minicube.fits   0.0862    0.0862 ...     7111.14            1.25               --          --          --              --         
         3 63.3552677 10.4657851 muselet      2.1 minicube.fits   0.0864    0.0864 ...     7112.39            1.25            332.9        30.3     6843.64            1.25         
         5 63.3567159 10.4656965 muselet      2.1 minicube.fits   0.0862    0.0863 ...     7308.64            1.25   [SII]2      --          --          --              --         
         7 63.3559897 10.4646136 muselet      2.1 minicube.fits       --        -- ...          --              --               --          --          --              --         
         8 63.3549327 10.4656274 muselet      2.1 minicube.fits       --        -- ...          --              --               --          --          --              --         

  In [9]: cat.colnames
  Out[9]: 
  ['ID',
   'RA',
   'DEC',
   'ORIGIN',
   'ORIGIN_V',
   'CUBE',
   'Z_EMI',
   'Z_EMI_MAX',
   'Z_EMI_MIN',
   'FLUX001',
   'FLUX_ERR001',
   'LBDA_OBS001',
   'LBDA_OBS_ERR001',
   'LINE001',
   'FLUX002',
   'FLUX_ERR002',
   'LBDA_OBS002',
   'LBDA_OBS_ERR002',
   'LINE002',
   'FLUX003',
   'FLUX_ERR003',
   'LBDA_OBS003',
   'LBDA_OBS_ERR003',
   'LINE003',
   'FLUX004',
   'FLUX_ERR004',
   'LBDA_OBS004',
   'LBDA_OBS_ERR004',
   'LINE004',
   'FLUX005',
   'FLUX_ERR005',
   'LBDA_OBS005',
   'LBDA_OBS_ERR005',
   'LINE005',
   'FLUX006',
   'FLUX_ERR006',
   'LBDA_OBS006',
   'LBDA_OBS_ERR006',
   'LINE006']

Then, we visualize these sources on our white image by using `~mpdaf.sdetect.Catalog.plot_id`::

  In [10]: from mpdaf.obj import Cube

  In [11]: cube = Cube('../data/sdetect/minicube.fits')

  In [12]: ima = cube.sum(axis=0)

  In [13]: fig = plt.figure()

  In [14]: ax = fig.add_subplot(1,1,1)

  In [15]: ima.plot()
  Out[15]: <matplotlib.image.AxesImage at 0x7f8b61288490>

  In [16]: cat.plot_id(ax, ima.wcs)

.. image::  _static/sources/catalog_id.png

`~mpdaf.sdetect.Catalog.edgedist` returns the smallest distance of all catalog sources center to the
edge of the WCS of the given image::

  In [17]: cat.edgedist(ima.wcs)
  Out[17]: 
  array([ 2.2983 ,  0.4317 ,  2.83236,  0.2    ,  2.706  ,  0.1674 ,
          0.11268,  1.52012]) 
	      
`~mpdaf.sdetect.Catalog.select` selects all sources from catalog which are inside the WCS of an image.
We will test it on the sub-image::

  In [18]: ima2 = ima[10:25, 15:30]

  In [19]: cat2 = cat.select(ima2.wcs)

  In [20]: len(cat2)
  Out[20]: 1

  In [21]: cat2
  Out[21]: 
  <Catalog masked=True length=1>
     ID        RA        DEC      ORIGIN ORIGIN_V      CUBE      Z_EMI   Z_EMI_MAX ... LBDA_OBS005 LBDA_OBS_ERR005 LINE005  FLUX006 FLUX_ERR006 LBDA_OBS006 LBDA_OBS_ERR006 LINE006 
  unitless    deg        deg                                    unitless  unitless ...   Angstrom      Angstrom    unitless                       Angstrom      Angstrom    unitless
   int64    float64    float64     str7  float64      str13     float64   float64  ...   float64       float64      str20   float64   float64     float64       float64      str20  
  -------- ---------- ---------- ------- -------- ------------- -------- --------- ... ----------- --------------- -------- ------- ----------- ----------- --------------- --------
         3 63.3552677 10.4657851 muselet      2.1 minicube.fits   0.0864    0.0864 ...     7112.39            1.25            332.9        30.3     6843.64            1.25         

Of course, if we `~mpdaf.sdetect.Catalog.match` this second catalog with the first, the result is evident::

  In [22]: cat3 = cat.match(cat2)
  [DEBUG] Cat1 Nelt 8 Matched 1 Not Matched 7
  [DEBUG] Cat2 Nelt 1 Matched 1 Not Matched 0
