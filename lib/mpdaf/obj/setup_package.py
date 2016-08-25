from distutils.extension import Extension


def get_extensions():
    return Extension('merging', ['../../../src/tools.c', 'merging.pyx'],
                     include_dirs=['numpy'])
