import os
import subprocess
import sys
from distutils.extension import Extension

PY2 = sys.version_info[0] == 2


def options(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}

    for package in packages:
        try:
            out = subprocess.check_output(['pkg-config', '--modversion',
                                           package])
        except subprocess.CalledProcessError:
            msg = "package '{}' not found.".format(package)
            print(msg)
            raise Exception(msg)
        else:
            if not PY2:
                out = out.decode('utf8')
            print('Found {} {}'.format(package, out))

    for token in subprocess.check_output(["pkg-config", "--libs", "--cflags",
                                          ' '.join(packages)]).split():
        if not PY2:
            token = token.decode('utf8')
        if token[:2] in flag_map:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else:  # throw others to extra_link_args
            kw.setdefault('extra_link_args', []).append(token)

    kw.setdefault('libraries', []).append('m')

    # Use OpenMP if directed or not on a Mac
    if os.environ.get('USEOPENMP') or not sys.platform.startswith('darwin'):
        kw.setdefault('extra_link_args', []).append('-lgomp')
        kw.setdefault('extra_compile_args', []).append('-fopenmp')
    else:
        print("Unable to find OPENMP")

    for k, v in kw.items():  # remove duplicated
        kw[k] = list(set(v))
    return kw


def get_extensions():
    return [Extension('tools._ctools', [
        'src/tools.c', 'src/subtract_slice_median.c', 'src/merging.c'],
        **options('cfitsio'))]
