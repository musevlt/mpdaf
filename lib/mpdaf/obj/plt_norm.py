import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cbook as cbook

class ArcsinhNorm(Normalize):
    """
    Normalize a given value to arcsinh scale
    """
    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin==vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                val = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                mask=mask)
            else:
                val = result
                
            result = (val-vmin) * (1.0/(vmax-vmin))
            midpoint = -0.033
            result = np.ma.arcsinh(result/midpoint) / np.ma.arcsinh(1./midpoint)
 
        if is_scalar:
            result = result[0]
 
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = np.ma.asarray(value)
        else:
            val = value
 
        midpoint = -0.033
        val = midpoint * np.ma.sinh(val*np.ma.arcsinh(1./midpoint))
        return vmin + val * (vmax - vmin)

class PowerNorm(Normalize):
    """
    Normalize a given value to power scale
    """
    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin==vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                val = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                mask=mask)
            else:
                val = result
                
            result = (val-vmin) * (1.0/(vmax-vmin))
            result = np.ma.power(result, 2)
 
        if is_scalar:
            result = result[0]
 
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = np.ma.asarray(value)
        else:
            val = value
 
        val = np.ma.power(val, (1./2))
        return vmin + val * (vmax - vmin)


class SqrtNorm(Normalize):
    """
    Normalize a given value to square root scale
    """
    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin==vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                val = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                mask=mask)
            else:
                val = result
                
            result = (val-vmin) * (1.0/(vmax-vmin))
            result = np.ma.sqrt(result)
 
        if is_scalar:
            result = result[0]
 
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = np.ma.asarray(value)
        else:
            val = value
 
        val = val * val
        return vmin + val * (vmax - vmin)
               
class LogNorm(Normalize):
    """
    Normalize a given value to the 0-1 range on a log scale
    """
    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        result = np.ma.masked_less_equal(result, 0, copy=False)

        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin<=0:
            raise ValueError("values must all be positive")
        elif vmin==vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                val = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                mask=mask)
            midpoint = 0.05
            result = np.ma.log10((result/midpoint) + 1.) / np.ma.log10((1./midpoint) + 1.)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = np.ma.asarray(value)
        else:
            val = value
        midpoint = 0.05
        val = midpoint * (np.ma.power(10., (val*np.ma.log10(1./midpoint+1.))) - 1.)
        
        return val

    def autoscale(self, A):
        '''
        Set *vmin*, *vmax* to min, max of *A*.
        '''
        A = np.ma.masked_less_equal(A, 0, copy=False)
        self.vmin = np.ma.min(A)
        self.vmax = np.ma.max(A)

    def autoscale_None(self, A):
        ' autoscale only None-valued vmin or vmax'
        if self.vmin is not None and self.vmax is not None:
            return
        A = np.ma.masked_less_equal(A, 0, copy=False)
        if self.vmin is None:
            self.vmin = np.ma.min(A)
        if self.vmax is None:
            self.vmax = np.ma.max(A)