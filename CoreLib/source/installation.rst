Download and install mpdaf
**************************


Download the code
=================

Trac system
-----------

The Trac repository browser `Browse Source <http://urania1.univ-lyon1.fr/mpdaf/browser>`_ can be used to navigate through the directory structure.

Or you can browse specific revisions in the wiki page `CoreLib <http://urania1.univ-lyon1.fr/mpdaf/wiki/WikiCoreLib>`_.


Git repository
--------------

If you want to get a local copy of the mpdaf project, it is better to clone it with git.

To clone the current development branch, you simply run the *git clone [url]* command::

  git clone http://urania1.univ-lyon1.fr/git/mpdaf


By default, Git will create a directory labelled mpdaf. If you want something different, you can just put it at the end of the command, after the URL. 


mpdaf contains large packages (`fusion <user_manual_fusion.html>`_ and quickViz) and the user has the choice to download or not download them. *submodule* git option is used

After the *git clone* command, the submodules directories are there, but they're empty. Pulling down the submodules is a two-step process.

First select the submodules that you want used. Now use *git submodule update*::

  /mpdaf$ git submodule init lib/mpdaf/fusion
  /mpdaf$ git submodule init lib/mpdaf/quickViz
  /mpdaf$ git submodule update


Then, you use *git pull* command to bring your repository up to date::

  /mpdaf$ git pull
  /mpdaf$ git submodule update


Prerequisites
=============

The various software required are:

 * Python (version 2.6 or 2.7)
 * IPython
 * setuptools
 * numpy (version 1.6.2 or above)
 * scipy (version 0.10.1 or above)
 * matplotlib (version 1.1.0 or above)
 * pyfits (version 3.0 or above)
 * pywcs (version 1.11-4.7 or above)
 * nose


Installation
============

To install the mpdaf package, you first run the *setup.py build* command to build everything needed to install::

  /mpdaf$ python setup.py build


Then, you lof as root and install everything from build directory::


  root:/mpdaf$ python setup.py install


setup.py informs you that the fusion package is not found. But it's just a warning, it's not blocking and you can continue to install mpdaf.

To install the fusion submodule, log as root and run the *setup.py fusion* command::

  root:/mpdaf$ python setup.py fusion



Unit tests
==========

The command *setup.py test* runs unit tests after in-place build::

  /mpdaf$ python setup.py test
