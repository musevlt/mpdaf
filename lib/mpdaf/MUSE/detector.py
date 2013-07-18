#!/usr/bin/env python
""" detector.py detector performance evaluation"""
import numpy as np
import pyfits
import multiprocessing
import datetime
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import prettytable
from mpdaf.drs import RawFile
from mydrs.calib import plot_qc_quads
from mydrs.util import get_valid_channels

class BIAS(object):
    """This class perform detector bias performance evaluation
    biaslist: list of raw bias exposure
    """
    def __init__(self, biaslist, gain=1.15, size=100,
                 quad=[[1000,1000],[3000,1000],[3000,3000],[1000,3000]]):
            self.biaslist = biaslist
            self.gain = gain
            self.size = size
            self.quad = quad
            
    def comp_ron(self, chanlist=None, nsigma=10, quiet=False, ncpu=0):
        """ compute RON from the list of exposures
        chanlist: list of channel, if None all existing channels are used, if 0 use 24 channels
        nsigma: rejection for bad pixel with respect to median value and standard deviation
        quiet: if True do not print message
        ncpu: nomber of cpu to use, if 0 use all available cpus
        """
        if chanlist is None:
            chanlist = get_valid_channels(self.biaslist[0])
        elif chanlist == 0:
            chanlist = range(1,25) 
        if not quiet:
            print 'Computing RON from %d exposures and %d channels'%(len(self.biaslist), len(chanlist))
        if ncpu == 0:
            ncpu = min(multiprocessing.cpu_count(), len(chanlist))
        if ncpu > 0:
            pool = multiprocessing.Pool(processes = ncpu)
            if not quiet:
                print 'Parallel execution [%d cores]'%(ncpu)
                
        plist = [[self.biaslist, chan, self.gain, self.size, nsigma, quiet, self.quad] for chan in chanlist]
        pres = pool.map(_compute_ron, plist)
        chan = []
        ronval = []
        ronlist = []
        for p in pres:
            chan.append(p[0])
            ronval.append(p[1])
            ronlist.append(p[2])
        chan = np.array(chan)
        ronval = np.array(ronval)
        ronlist = np.array(ronlist)
        ksort = chan.argsort()
        chan = chan[ksort]
        ronval = ronval[ksort]
        ronlist = ronlist[ksort]
        self.ron = {'chan':chan, 'ron':ronlist, 'ronval':ronval}
        
    def print_ron(self):
        """ Print RON values for all channels
        """
        if not hasattr(self, 'ron'):
            raise ValueError, 'Run bias.comp_ron() first'
        ptable =  prettytable.PrettyTable()
        ron = self.ron
        ptable.add_column('CH', ron['chan'], align='r')
        ptable.add_column('Q1', np.round(ron['ronval'][:,0], 2), align='l')
        ptable.add_column('Q2', np.round(ron['ronval'][:,1], 2), align='l') 
        ptable.add_column('Q3', np.round(ron['ronval'][:,2], 2), align='l') 
        ptable.add_column('Q4', np.round(ron['ronval'][:,3], 2), align='l') 
        print ptable
        
    def plot_ron(self):
        """ Plot RON values for all channels
        """
        if not hasattr(self, 'ron'):
            raise ValueError, 'Run bias.comp_ron() first'
        ron = self.ron
        plot_qc_quads(ron['chan'], ron['ronval'], spec=3)
        axis = plt.axis()
        plt.axis((0,25,axis[2],axis[3]*1.1))        
        plt.xlabel('Channel', fontsize='medium')
        plt.ylabel('Readout Noise (e-)', fontsize='medium') 
        
    def disp_chan(self, expnum, nochan):
        """ display a single exposure and the location of the RON computation
        expnum: index of exposure from the biaslist
        nochan: channel number
        """        
        raw = RawFile(self.biaslist[expnum])
        chan = raw[nochan]
        ima = chan.get_trimmed_image()
        ima.plot()
        for box in self.quad:
            _box_plot(box, self.size)
        plt.title('File %s Chan %02d'%(self.biaslist[expnum], nochan))
        
    def disp_quad(self, expnum, nochan):
        """ Display quadrants for a given exposure and channel
        expnum: index of exposure from the biaslist
        nochan: channel number
        """
        raw = RawFile(self.biaslist[expnum])
        chan = raw[nochan]
        ima = chan.get_trimmed_image()
        del raw
        region = [ima[p-self.size:p+self.size,q-self.size:q+self.size] for p,q in self.quad] 
        for k,q in zip([1,2,3,4],[4,3,1,2]):
            plt.subplot(2,2,k)
            region[q-1].plot(cmap='gray')
            plt.axis('off')
            plt.title('Q%d'%(q)) 
            
    def plot_seq_ron(self, nochan):
        """ Plot the sequence of computed RON for a given channel
        nochan: channel number
        """        
        if not hasattr(self, 'ron'):
            raise ValueError, 'Run bias.comp_ron() first'
        k = np.where(self.ron['chan'] == nochan)[0][0]
        ron = self.ron['ron'][k,:,:]
        plt.plot(ron[:,0], '-ob', label='Q1')
        plt.plot(ron[:,1], '-og', label='Q2')
        plt.plot(ron[:,2], '-or', label='Q3')
        plt.plot(ron[:,3], '-ok', label='Q4')
        plt.title('Channel %02d'%(nochan))
        plt.legend(numpoints=1)
        
            
def _box_plot(box, size):
    x0,y0 = box
    x1 = x0 - size
    x2 = x0 + size
    y1 = y0 - size
    y2 = y0 + size
    plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], '-k')
        
        
def std_medclip(data, nsigma):
    vmed = np.median(data)
    vstd = np.std(data)
    k = np.where(np.abs(data - vmed)<nsigma*vstd)
    return np.std(data[k])

def _compute_ron(arglist):
    biaslist, nochan, gain, size, nsigma, quiet, quad = arglist
    ron = []
    for bname in biaslist:
        raw = RawFile(bname)
        chan = raw[nochan]
        ima = chan.get_trimmed_image()
        del raw
        region = [ima[p-size:p+size,q-size:q+size].data * gain for p,q in quad]
        ronval = [std_medclip(data, nsigma) for data in region]
        if not quiet:
            print '\tFile %s RON %5.1f %5.1f %5.1f %5.1f'%(bname, ronval[0], ronval[1], ronval[2], ronval[3])
        ron.append(ronval)
        del region
        del ima
    ron = np.array(ron)
    ronval = np.median(ron, axis=0) 
    print 'Chan %02d RON : %5.1f %5.1f %5.1f %5.1f'%(nochan, ronval[0], ronval[1], ronval[2], ronval[3])
    return [nochan,ronval,ron]


        
        
if __name__ == "__main__":
    from optparse import OptionParser
    import sys
    from glob import glob
    from os import chdir
    
    #parser = OptionParser(usage="%prog file\n", version="%prog " + __version__)
    #parser.add_option("-m", "--method", dest="method", 
                      #help="Method: average|median|sum [average]", default='average') 
    #parser.add_option("-r", "--reject", dest="reject", 
                      #help="Reject Method: none|minmax|ccdclip|crreject|sigclip|avsigclip|pclip [avsigclip]",
                      #default='avsigclip') 
    #parser.add_option("--minmax_rej", dest="minmax_rej", 
                      #help="Minmax algo: nlow,nhigh,nkeep. [0,0,1]",
                      #default='0,0,1')
    #parser.add_option("--clip_rej", dest="clip_rej", 
                      #help="Clipping algo: mclip (1/0),lsigma,hsigma [1,3,3]", default='1,3,3')    
    
    chdir('/Users/rolandbacon/ait/NGC-retrofit/18July-allsources-stopped')    
    biaslist = glob('raw/*.fits')
    bias = BIAS(biaslist)
    bias.ron([14,5,6])
    bias.print_ron()
    #plot_qc_quads(res['chan'], res['ronval'])