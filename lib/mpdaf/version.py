__version__ = '1.2'
__date__ = '2016/01/13'

try:
    from ._githash import __githash__, __dev_value__
    __version__ += __dev_value__
except Exception:
    pass
