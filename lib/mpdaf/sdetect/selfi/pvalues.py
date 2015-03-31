"""
Created on Mon Jan 27 13:20:55 2014
    
@author: meillice
"""

import astropy.io.fits as pyfits


class p_values(object):
    """ 
        
        Parameters
        ----------
        val         : string
                  filename
    cdf_maxTest : string
              filename
        
        Attributes
        ----------
        a     : array
        
        p_val : list
        
        """
    
    def __init__(self, val, cdf_maxTest):
        """ 
            
        Parameters
        ----------
        val         : string
                      filename
        cdf_maxTest : string
                      filename
        """
        f = pyfits.open(val)
        self.a = f[0].data
        f.close()
        f = pyfits.open(cdf_maxTest)
        b = f[0].data
        f.close()
        c = 1. - b
        self.p_val = list(c)

    def p_val_maxTest(self,value):
        """ Evaluates the p-value corresponding to a test value
        
          Parameters
          ----------
          value : float
                  Value of the test
        """
        val = list(self.a)
        val.append(value)
        val.sort()
        index = val.index(value)
        if index > 0:
            result = self.p_val[index-1]
        else:
            result = self.p_val[0]
        return result
    