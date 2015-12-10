__version__ = '1.2.dev'
__date__ = '2015/11/05'

try:
    from ._githash import __githash__, __dev_value__
    __version__ += __dev_value__
except Exception:
    pass
