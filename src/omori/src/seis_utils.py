#!usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
import random
import datetime

font = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)

## Read catalog using pandas dataframe and convert to numpy array
def read_cat(filename):
    pdcat = pd.read_csv(filename, dtype='str')
    c    = pdcat.to_numpy()
    time = c[:,0]
    lat  = c[:,1].astype('float')
    lon  = c[:,2].astype('float')
    dep  = c[:,3].astype('float')
    mag  = c[:,4].astype('float')
    evid = c[:,11]
    obdt = []
    pydt = []
    relt = []
    for i in range(len(time)):
        obdt.append(UTC(time[i]))
        pydt.append(UTC(time[i]).datetime)
        relt.append((UTC(time[i]) - UTC(time[0])) / 86400)
    return (evid, obdt, pydt, relt, lat, lon, dep, mag)


def read_cat_hauk(filename):
    c = np.genfromtxt(filename, dtype='str')
    yr = c[:,0].astype('int')
    mo = c[:,1].astype('int')
    dy = c[:,2].astype('int')
    hr = c[:,3].astype('int')
    mi = c[:,4].astype('int')
    sc = c[:,5].astype('float')
    evid = c[:,6].astype('str')
    lon  = c[:,7].astype('float')
    lat  = c[:,8].astype('float')
    dep  = c[:,9].astype('float')
    mag  = c[:,10].astype('float')
    obdt = []
    pydt = []
    relt = []
    for i in range(len(evid)):
        tstr = '%02d%02d%02d%02d%02d%5.2f' % (yr[i],mo[i],dy[i],hr[i],mi[i],sc[i])
        obdt.append(UTC(tstr))
        pydt.append(UTC(tstr).datetime)
        relt.append((UTC(tstr) - obdt[0]) / 86400)
    return (evid, obdt, pydt, relt, lat, lon, dep, mag)


def read_cat_sep(filename):
    c = np.genfromtxt(filename, dtype='str')
    yr = c[:,0].astype('int')
    mo = c[:,1].astype('int')
    dy = c[:,2].astype('int')
    hr = c[:,3].astype('int')
    mi = c[:,4].astype('int')
    sc = c[:,5].astype('float') 
    lat  = c[:,6].astype('float')
    lon  = c[:,7].astype('float')
    dep  = c[:,8].astype('float')
    mag  = c[:,9].astype('float')
    evid = c[:,14]
    obdt = []
    pydt = []
    relt = []
    for i in range(len(evid)):
        tstr = '%02d%02d%02d%02d%02d%5.2f' % (yr[i],mo[i],dy[i],hr[i],mi[i],sc[i])
        obdt.append(UTC(tstr))
        pydt.append(UTC(tstr).datetime)
        relt.append((UTC(tstr) - obdt[0]) / 86400)
    return (evid, obdt, pydt, relt, lat, lon, dep, mag)


def read_cat_savesep(filename):
    pdcat = pd.read_csv(filename, dtype='str', sep=" |/|:|,", engine="python")
    c     = pdcat.to_numpy()
    np.savetxt('test.out', c, fmt='%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s')


# MAXC method of determining magnitude of completeness
def maxc_Mc(mag_arr, plot='no', title=''):
    bin_size = 0.05
    fc = 'lightskyblue'
    ec = 'k'
    k, bin_edges = np.histogram(mag_arr,np.arange(-1,10,bin_size))
    centers      = bin_edges[:-1] + np.diff(bin_edges)[0]/2
    correction   = 0.20
    Mc = centers[np.argmax(k)] + correction
    xloc  = 0.50
    yloc1 = 0.90
    yloc2 = 0.70
    yloc3 = 0.55

    if plot == 'yes':
        fig ,ax = plt.subplots(figsize=[18,5], ncols=2)
        ax[0].bar(bin_edges[:-1], k, width=bin_size, fc=fc, ec=ec)
        ax[0].axvline(x=Mc, c='k', ls='--', lw=3)
        ax[0].set_xlim(-0.5,4)
        ax[0].text(xloc, yloc1, 'Mc = {:.3f}'.format(Mc), transform=ax[0].transAxes)
        ax[0].text(xloc, yloc2, r'N($M \geq Mc$) = {:d}'.format(len(mag_arr[mag_arr>=Mc])), transform=ax[0].transAxes)
        ax[0].text(xloc, yloc3, r'N($M < Mc$) = {:d}'.format(len(mag_arr[mag_arr<Mc])), transform=ax[0].transAxes)        
        ax[0].set_xlabel('Magnitude')
        ax[0].set_ylabel('# events')
        ax[0].text(0.35, 1.03, title, fontsize=22, transform=ax[0].transAxes)                

        ax[1].bar(bin_edges[:-1], 100*(1-np.cumsum(k)/np.cumsum(k)[-1]), width=bin_size, fc=fc, ec=ec)
        ax[1].axvline(x=Mc, c='k', ls='--', lw=3)
        ax[1].set_xlim(-0.5,4)
        ax[1].text(xloc, yloc1, 'Mc = {:.3f}'.format(Mc), transform=ax[1].transAxes)
        ax[1].text(xloc, yloc2, r'N($M \geq Mc$) = {:d}'.format(len(mag_arr[mag_arr>=Mc])), transform=ax[1].transAxes)
        ax[1].text(xloc, yloc3, r'N($M < Mc$) = {:d}'.format(len(mag_arr[mag_arr<Mc])), transform=ax[1].transAxes)          
        ax[1].set_xlabel('Magnitude')
        ax[1].set_ylabel('Cumulaive % events')
        ax[1].text(0.35, 1.03, title, fontsize=22, transform=ax[1].transAxes)        
        plt.show()
        
    elif plot == 'no':
        pass
    
    else:
        print('Plot keyword is either "yes" or "no"')
    
    return Mc  


def epoch_Mc(mag, obspyDT, Nt=10, plot='no', title=''):
    Ndt = (obspyDT[-1]-obspyDT[0])/Nt

    epochs = []
    for i in range(int(Nt)):
        epochs.append(np.where(np.array(obspyDT)>=obspyDT[0]+Ndt*i)[0][0])

    Mcs = []
    for i in range(Nt):
        if i==Nt-1:
            sub_mag = mag[epochs[i]:-1]
        else:
            sub_mag = mag[epochs[i]:epochs[i+1]]
        Mcs.append(maxc_Mc(sub_mag, plot=plot, title=title)) 
    return epochs, Ndt, Mcs