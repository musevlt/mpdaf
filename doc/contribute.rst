************
Contributing
************

This page gives some advices on how to contribute code to MPDAF. Astropy has
a `well-detailed documentation`_ on contributing to an open-source project
using Git, which is also mostly applicable to MPDAF, except for a few
differences: MPDAF uses a Gitlab instance instead of Github, and the tests
runner is `nose`_ instead of py.test.

Getting an account
------------------

The first pre-requisite is to have an account on the `git-cral`_ server. If
you don't have one yet, please send email to
`mpdaf-support@osulistes.univ-lyon1.fr
<mailto:mpdaf-support@osulistes.univ-lyon1.fr?subject=Account%20creation>`_.

Installation
------------

You need to install the development version of MPDAF from the git repository,
in your environment::

    git clone https://git-cral.univ-lyon1.fr/MUSE/mpdaf.git

There are many options to install, but it is recommended to use a `virtual
environment`_ (virtualenv, or Conda's environments) to avoid conflicts with
your regular install.

Then, you can install MPDAF. Using the development mode makes it easier to test
changes, without having to reinstall after each change::

    python setup.py develop


Git workflow
------------

A classic Git development workflow is used, using branches and merge requests
(the equivalent of Github's Pull requests).

Make sure to start from an up-to-date master::

    git checkout master
    git pull

Then create a branch and work on it::

    git checkout -b my-new-feature
    # edit files ...
    git commit

And push your branch to the server::

    git push --set-upstream origin my-new-feature

You can now create a `Merge request`_.

Unit tests
----------

Unit tests are run automatically on the server, after each push to a branch,
and the `build status`_ is shown on merge requests. It is of course strongly
recommended to add a few tests to test the new feature you developed.

It is also a good idea to run the tests locally before pushing to the server,
to find errors more quickly and avoid running too many builds on the server.

To run the tests, you need to install `nose`_::

    pip install nose

And run::

    python setup.py test

It is also possible to run tests on multiple Python versions with `tox`_::

    pip install tox



.. _build status: https://git-cral.univ-lyon1.fr/MUSE/mpdaf/builds
.. _git-cral: https://git-cral.univ-lyon1.fr
.. _Merge request: https://git-cral.univ-lyon1.fr/MUSE/mpdaf/merge_requests
.. _nose: https://nose.readthedocs.io/en/latest/
.. _tox: http://tox.readthedocs.io/en/stable/
.. _virtual environment: http://docs.astropy.org/en/latest/development/workflow/virtual_pythons.html
.. _well-detailed documentation: http://docs.astropy.org/en/latest/development/workflow/development_workflow.html

