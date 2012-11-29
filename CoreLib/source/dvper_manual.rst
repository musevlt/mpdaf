Develop an user package with MPDAF
**********************************

This page gives some directives to develop an user package in the MPDAF environment. The user developed packages are available for the consortium via an user library, labeled UserLib.


Coding
======



Coding convention
-----------------

  * MPDAF package should follow the Python coding conventions PEP8 (`<http://www.python.org/dev/peps/pep-0008>`_ )
  
  * MPDAF package should be compatible with Python 2.6 or later (including 3.x version)
  
  * MPDAF package should run on the following 64 bits system: Linux and Mac OS X
  
  * MPDAF should be able to make an efficient use of multi-core CPU
  
  * MPDAF shall be able to run on computer with limited memory and still be able to play with large datacube
  
  * MPDAF shall be able to handle the following units:
  
    * Wavelength: nm, A
  
    * WCS: absolute coordinate in ra,dec deg,mn,sec, or relative coordinate in arcsec
  
    * Flux: erg/s/cm2/A
  
  * Basic and advanced operation for the MPDAF objects should be developed using numpy and scipy


Directory structure
-------------------

MPDAF is structured in 5 repositories. The first four repositories are dedicated to the CoreLib library:

  * *lib* contains code (Python/C++, Java)
  * *tests* contains unit tests
  * *data* contains tests files
  * *doc* contains HTML documentation

The last repository, labeled *mpdaf_user*, is dedicated to the MUSE consortium to add new scripts.

In it, each new repository should correspond to a user package. It should be divided in 4 parts: *lib*, *tests*, *data* and *doc*::

  mpdaf/
	lib/
	tests/
	data/
	doc/
  mpdaf_user/
	my_package/
		lib/
		tests/
		data/
		doc/

Python packages and modules
---------------------------

Python packages are used to structure Python code in modules.  MPDAF project contains two top level packages:

  * *mpdaf* contains the different Python packages defining the CoreLib library
  
  * *mpdaf_user* contains the different Python packages defining the UserLib library

Python packages are organized in terms of a hierarchical file system. For example, the obj directory describes the mpdaf.obj package of the CoreLib library::

  mpdaf/
	lib/
		obj/                         # Top-level package
      			__init__.py          # Initialize the top_level package
      			coords.py	     # Code about world coordinates
			spectrum.py	     # Code about spectrum
			image.py	     # Code about image
			cube.py		     # Code about cube

The *__init__.py* files is required to make Python treat the directory as containing packages. Such an *__init__.py* file is presented bellow::

  __version__  = '1.0.2'
  __date__     = '2012/11/19 16:47'
  from coords import WCS
  from coords import WaveCoord
  from spectrum import Spectrum
  from image import Image
  from cube import Cube
  …

Users of the package should be able to import individual modules from the package, for example::

  > from mpdaf.obj import Image


The package structure should be the same for an user package::

  mpdaf/
	...
  mpdaf_user/
	__init__.py				# import my_package
	my_package/
		__init__.py			# empty file
		lib/
			__init__.py		# from code import Test
			code.py
		tests/
		data/
		doc/


The module Test should be imported with the command::

  > from mpdaf_user.my_package import Test


See `<http://docs.python.org/2/tutorial/modules.html>`_ for more information.


Installation
------------

MPDAF is installed like a Python standard automatic package . *Python Distribution Utilities (Distutils)* is used.

If the structure described in part `Python packages and modules`_ is followed, user packages will be automatically installed by the installation script of MPDAF (see :ref:`MPDAF installation <installation-label>`)
  

See `<http://docs.python.org/2/distutils/setupscript.html>`_ for more information on *Distutils*.


Git repository
==============

The Git version control system is used to handle the MPDAF project. MPDAF git server is located on urania1 machine at Lyon.

Users do not need an urania1 account to download the MPDAF package from the server, and should use git through the http protocol::

  > git clone http://urania1.univ-lyon1.fr/git/mpdaf
  
  
Users who want to contribute code to MPDAF should develop their packages separately but in the MPDAF environment. We want to be able to treat the two projects as separate yet still be able to use one from within the other. Git addresses this issue using submodules. Submodules allow to keep a Git repository as a subdirectory of another Git repository. 
As described in `Python packages and modules`_, *mpdaf_user* repository is dedicated to the MUSE consortium to add new scripts. So, user packages should be stored as a Git submodule in the *mpdaf_user* repository. The user repository will be cloned into the MPDAF project and users will keep their commits separated.
  

Users who want to develop a user package and commit their own changes to the server need an urania1 account. In that case the current development branch of the user package should be cloned through the ssh protocol::

  > git clone urania1.univ-lyon1.fr:/git/mpdaf_mypackage
  
Then the *git add* command could be used to schedule the addition of an individual file to the next commit::

  > mpdaf_mypackage$ git add [file name]
  

The commit is then done with the following command::

  > mpdaf_mypackagef$ git commit -m "This is the message describing the commit"
  
The git push command is used to send changes from the user local repository to the repository on urania1::
		
  > mpdaf_mypackage$ git push origin


Stable versions of the user package will be added on the UserLib library of MPDAF by CRAL.

Note that after a Git clone of MPDAF, the submodule directories are there, but they are empty. Pulling down the submodules is a two-step process. 

First select the submodules that you want to be used. Now use git submodule update:: 
		
  > mpdaf$ git submodule init mpdaf_user/my_package
  > mpdaf$ git submodule update
  
Then, use the git pull command to bring your repository up to date::

  > mpdaf$ git pull
  > mpdaf$ git submodule update
  
  
See `<http://www.kernel.org/pub/software/scm/git/docs/user-manual.html>`_ for more information on *Git*.


Units tests
===========

Unitary testing of MPDAF is done using the Python tool *nose*. It automatically finds and executes tests (`<https://nose.readthedocs.org/en/latest/>`_).

Python tests are structured as Python code. For example, the obj directory containing the test about the obj package is the following::

  obj/                          
	test_coords.py		# Test about world coordinates
	test_spectrum.py	# Test about spectrum
	test_image.py		# Test about image
	test_cube.py		# Test about cube


MPDAF tests are divided in two parts according to the computation time/memory use:

  * general unit tests that will be run each time. The corresponding data is stored in the data repository. The data directory is also structured by package.
  
  * huge tests that will be only run by developers. The data are stored in an independent git repository (urania1.univ-lyon1.fr:/git/mpdaf_data)

A decorator is used (`<https://nose.readthedocs.org/en/latest/plugins/attrib.html>`_) to split the tests::

  from nose.plugins.attrib import attr
  
  @attr(speed='slow')
  def test_big():
      # test ...
      
  @attr(speed='fast')
  def test():
      # test ...
      
These tests could be run with::

  > nosetests -v -a speed=slow/fast


Documentation
=============

User manual with Sphinx
-----------------------

MPDAF should be documented for the user. HTML documentation is available on the folder *mpdaf/doc*. MPDAF user manual is created using the *sphinx* tool which has excellent facilities for the documentation of Python projects.

To update this documentation, clone the corresponding git repository with a ssh protocol::

  > git clone urania1.univ-lyon1.fr:/git/mpdaf_sphinx
  
Update source files and use *git add*, *git commit* and *git push* commands to send your changes from your working copy to the repository on urania1::

  mpdaf_sphinx> git add [file name]
  mpdaf_sphinx> git commit -m "This is the message describing the commit"
  mpdaf_sphinx> git push origin
  
The *CoreLib* repository contains the user manual of the CoreLib library. Documentation of user packages should be done in the *UserLib* repository.

In the *UserLib* folder, the *Source* directory will contains the '.rst' files of *sphinx* (see `<http://packages.python.org/an_example_pypi_project/sphinx.html>`_).

Convention names for these sphinx files are used in the MPDAF project:

  * [ClassName].rst generates a class documentation from docstrings in an automatic way. A class documentation is generated from docstrings like this::
  
	  .. autoclass:: mpdaf_user.my_package.class
	    :members: 

  * user_manual_[ClassName].rst contains overview of the class, tutorials and a list of methods. To link a method description to the corresponding lines in the class documentation use this::
  
	  :func:`mpdaf_user.my_package.class.method <mpdaf_user.my_package.class.method>`

  * [PackageName].rst describes the package. This page gives a link to the different user manual pages of this package.

The *UserLib* directory contains a Makefile. The HTML documentation is generated with the *make html* command.


Web interface
-------------

MPDAF is available through a web interface for software distribution (limited to the consortium) and bug/problem reporting
`<http://urania1.univ-lyon1.fr/mpdaf/login>`_

UserLib wiki page should describe developed packages available to the consortium.