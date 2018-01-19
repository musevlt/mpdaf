from distutils.extension import Extension


def get_extensions():
    return [Extension('mpdaf.obj.merging',
                      ['src/tools.c', 'lib/mpdaf/obj/merging.pyx'],
                      include_dirs=['numpy'])]
