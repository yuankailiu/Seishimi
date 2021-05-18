#!/usr/lib/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 9 2021

@author: ykliu
"""
#%% import necessary packages

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
import random
import datetime
import scipy
from scipy.optimize import minimize
from scipy import linalg
from scipy.special import factorial as fac
from geopy.distance import geodesic
import src.seis_utils as seis
import src.omori_utils as omori

import emcee
import corner 

matplotlib.rcParams.update({'font.size': 22})

fstr = 'pr'  # head string of saved files

# covered time period
t_start = UTC('19800101')
t_termi = UTC('20210101')

if fstr=='pk':
    titl = 'Parkfield M6.0'
    t_termi = UTC('20100101')
    cat = seis.read_cat('PKcat.txt')
elif fstr=='ss':
    titl = 'San Simeon M6.6'
    cat = seis.read_cat('west_cat.txt')
elif fstr=='kh':
    titl = 'Kettleman Hills M6.1'
    t_termi = UTC('19860101')
    cat = seis.read_cat('east_cat.txt')
elif fstr=='lp':
    titl = 'Loma Prieta M6.9'
    t_start = UTC('19890801')
    #t_termi = UTC('20200101')
    cat = seis.read_cat('LPcat2.txt')
elif fstr=='rc':
    titl = 'Ridgecrest M7.1'
    cat = seis.read_cat_hauk('events.hauk')
elif fstr=='pr':
    titl = 'Prague M2.7'
    cat = seis.read_cat_sep('prague.txt')

#%% Read and present dataset

evid, obdt, pydt, relt, lat, lon, dep, mag = np.array(cat)

# MS info
id0 = obdt >= t_start
ms_id = np.argmax(mag[id0])
ms = {'id':evid[id0][ms_id], 'obdt':obdt[id0][ms_id], 'pydt':pydt[id0][ms_id], 'tAMS':relt[id0][ms_id], 
      'lat':lat[id0][ms_id], 'lon':lon[id0][ms_id], 'dep':dep[id0][ms_id], 'mag':mag[id0][ms_id]}

# save meta parameters
meta = {'t_start'   : t_start,
        't_termi'   : t_termi,
        'Mcut'      : 1.5,
        'rmax'      : 10**(0.25*ms['mag']-.22),  # max radius of influence (Gardner & Knopoff, 1967)
        'nbin'      : 100, 
        'c_bound'   : [1e-4,   2],
        'K_bound'   : [   2, 1e4],
        'p_bound'   : [  .2,  2],
        'c0'        : .5, 
        'K0'        : 50, 
        'p0'        : 1.1,
        'ylim'      : [1e-3, 1e5],
        'xlim'      : [1e-3, 1e3],
        'syn_c'     : 0.6,
        'syn_p'     : 1.3,
        'syn_tStart': 1e-2,
        'syn_tEnd'  : 1e3, 
        'syn_N'     : 4000,}


# get aftershocks within a radius
aR = []
for i in range(len(evid)):
    aR.append(geodesic((ms['lat'], ms['lon']),(lat[i], lon[i])).km)
aR = np.array(aR)
rd_id = aR <= meta['rmax']


# selections
mc_id  = mag  >= meta['Mcut']
as_id  = obdt >  ms['obdt']
end_id = obdt <  t_termi
rd_id  = aR   <= meta['rmax']
select = mc_id * as_id * end_id * rd_id
evid, obdt, pydt, relt, lat, lon, dep, mag = np.array(cat).T[select].T

print(' Mainshock magnitude: %.2f \n' % ms['mag'],
      'Mainshock time: %s \n'         % ms['obdt'],
      'Minimum magnitude: %.2f \n'    % meta['Mcut'],
      'Maximum radius: %.2f \n'       % meta['rmax'], 
      'Start time: %s \n'             % meta['t_start'],
      'End time: %s \n'               % meta['t_termi'],      
      '# events selected: %d \n'      % select.sum())


relt = relt.astype('float')
lat = lat.astype('float')
lon = lon.astype('float')
dep = dep.astype('float')
mag = mag.astype('float')
relt = relt - ms['tAMS']

size = [10,6]#[22,10]
plt.figure(figsize=size)
sc = plt.scatter(lat, pydt, s=100, c=dep, cmap='hot_r', vmin=0, vmax=20, ec='k', lw=0.4, alpha=0.5)
plt.scatter(ms['lat'], ms['pydt'], s=600, ec='k', fc='gold', marker='*')
plt.xlabel('Lat')
plt.ylabel('Time')
plt.colorbar(sc, label='Depth [km]', shrink=0.4, pad=-0.001)
plt.tight_layout()
plt.savefig('pics/%s_lt.png' % fstr, dpi=100)

plt.figure(figsize=size)
sc = plt.scatter(pydt, mag, s=100, fc='lightgrey', ec='k', lw=0.4)
plt.scatter(ms['pydt'], ms['mag'], s=600, ec='k', fc='gold', marker='*')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig('pics/%s_tm.png' % fstr, dpi=100)

#%% Run minimizer

# Choose dataset: 
data = '1'
if data == '0':
    # Synthetic dataset
    c = meta['syn_c']
    p = meta['syn_p']
    synt = omori_syn(c, p, meta['syn_tStart'], meta['syn_tEnd'], meta['syn_N'])
    otimes = np.array(sorted(synt))

elif data == '1':
    # Real dataset
    otimes = np.array(sorted(relt))


# Calc likelihood:
objFunc1 = lambda X: omori.ogata_logL(otimes, X)
objFunc2 = lambda X: omori.bayes_logL(otimes, X)

disp = 0
method = 'SLSQP'


# Ogata 1989: MLE
ogata_fit = scipy.optimize.minimize(objFunc1, np.array([meta['c0'], meta['K0'], meta['p0']]), \
            bounds=np.array([meta['c_bound'], meta['K_bound'], meta['p_bound']]), \
            tol = 1e-4, method=method, options={'disp': disp, 'maxiter':500})
print(ogata_fit)

# Holschneider et al., 2012: Bayesian
bayes_fit = scipy.optimize.minimize(objFunc2, np.array([meta['c0'], meta['p0']]), \
            bounds=np.array([meta['c_bound'], meta['p_bound']]), \
            tol = 1e-4, method=method, options={'disp': disp, 'maxiter':500})
finalL, K  = omori.bayes_getK(otimes, bayes_fit['x'])
print('\n',bayes_fit)
print('       K:',K)


meta['Ogata_fit'] = list(ogata_fit['x'])
meta['Bayes_fit'] = [bayes_fit['x'][0], K, bayes_fit['x'][1]]

meta

# %% Plot log-likelihood in a 2D parameter space
Cs = np.linspace(meta['c_bound'][0], meta['c_bound'][1])
Ps = np.linspace(meta['p_bound'][0], meta['p_bound'][1])

L = np.zeros([len(Cs), len(Ps)])
for i in range(len(Cs)):
    for j in range(len(Ps)):
        L[i,j] = objFunc2((Cs[i],Ps[j]))
L = L.T
        
plt.figure(figsize=[12,8])
im = plt.pcolormesh(Cs, Ps, L, cmap='jet_r', shading='auto')
plt.colorbar(im, label='-log(Likelihood)')
plt.contour(Cs, Ps, L, levels=100, colors='w')
plt.scatter(meta['Bayes_fit'][0], meta['Bayes_fit'][2], marker='*', s=600, ec='k', fc='w')
#plt.xscale('log')
plt.xlabel('c-value')
plt.ylabel('p-value')
plt.tight_layout()
plt.savefig('pics/%s_mlecp.png' % fstr, dpi=100)

# %% Plot fitting results and data
plot_res = 'both'

bins = np.logspace(np.log10(otimes[0]), np.log10(otimes[-1]), meta['nbin'])
count, bine = np.histogram(otimes, bins=bins)
bin_loc = (bine[1:] + bine[:-1]) / 2
occ_dens = count/np.diff(bins)


plt.figure(figsize=[10,8])
plt.scatter(bin_loc, occ_dens, s=100, ec='k', fc='lightgrey', label='%d events' % len(otimes))
plt.xscale('log')
plt.yscale('log')
plt.ylim(meta['ylim'])
#plt.xlim(meta['xlim'])
plt.xlabel('Days after mainshock')
plt.ylabel('# events/day')

lgdstr1 = 'Ogata {c,K,p}={%.2f, %.2f, %.2f}'
lgdstr2 = 'Holsch {c,K,p}={%.2f, %.2f, %.2f}'

if plot_res=='o':
    plt.plot(bin_loc, omori.omori(bin_loc, meta['Ogata_fit']), c='r', label=lgdstr1 % tuple(meta['Ogata_fit']))
elif plot_res=='h': 
    plt.plot(bin_loc, omori.omori(bin_loc, meta['Bayes_fit']), c='b', labe=lgdstr2 % tuple(meta['Bayes_fit']))
elif plot_res=='both':
    plt.plot(bin_loc, omori.omori(bin_loc, meta['Ogata_fit']), c='r', label=lgdstr1 % tuple(meta['Ogata_fit']))
    plt.plot(bin_loc, omori.omori(bin_loc, meta['Bayes_fit']), c='b', label=lgdstr2 % tuple(meta['Bayes_fit']))

plt.legend(loc='lower left')
plt.title('Aftershocks decay')
plt.tight_layout()
plt.savefig('pics/%s_mle.png' % fstr, dpi=100)


#%% Now try to fit with MCMC!

# define uniform rectangular prior 
def log_prior(theta):
    c, K, p = theta
    c1 ,c2 = meta['c_bound']
    K1 ,K2 = meta['K_bound']    
    p1 ,p2 = meta['p_bound']
    if c1 < c < c2 and K1 < K < K2 and p1 < p < p2:
        return 0.0
    return -np.inf

# define post prob = prior * likelihood
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp - objFunc1(theta)
   

# %% Run ensemble MCMC samplers

# initialize walkers in gaussian ball around minimized result 
pos0     = meta['Ogata_fit']
nwalkers = 32
ndim     = len(pos0)
pos      = pos0 + 1e-3 * np.random.randn(nwalkers, ndim)
sampler  = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 5000, progress=True);


# %% Check MCMC results

# Check out the chains
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ['c', 'K', 'p']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], 'k', alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel('# Step')
plt.tight_layout()
fig.savefig('pics/%s_mcmctrace.png' % fstr, dpi=100)


# check autocorrelation time and corner plot
tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

matplotlib.rcParams.update({'font.size': 13})
fig = corner.corner(flat_samples, labels=labels, 
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_kwargs={"fontsize": 16})
matplotlib.rcParams.update({'font.size': 22})
plt.tight_layout()
plt.savefig('pics/%s_mcmccorner.png' % fstr, dpi=100)

# %% Plot data against MCMC ensemble fits

plt.figure(figsize=[7,6])
plt.scatter(bin_loc, occ_dens, s=100, ec='k', fc='lightgrey', label='obs. %d events' % len(otimes))
plt.xscale('log')
plt.yscale('log')
plt.ylim(meta['ylim'])
#plt.xlim(meta['xlim'])
plt.xlabel('Days after mainshock')
plt.ylabel('# events / day')

sol_c = np.percentile(flat_samples[:, 0], [16, 50, 84])
qc = np.diff(sol_c)
sol_K = np.percentile(flat_samples[:, 1], [16, 50, 84])
qK = np.diff(sol_K)
sol_p = np.percentile(flat_samples[:, 2], [16, 50, 84])
qp = np.diff(sol_p)

inds = np.random.randint(len(flat_samples), size=200)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(bin_loc, omori.omori(bin_loc, sample), 'C1', alpha=0.1)
lgdstr1 = 'Best {c,K,p} =\n{%.2f, %.2f, %.2f}' % (sol_c[1],sol_K[1],sol_p[1])
plt.plot(bin_loc, omori.omori(bin_loc, (sol_c[1],sol_K[1],sol_p[1])), c='k', lw=2, label=lgdstr1)
plt.legend(loc='lower left')
plt.title(titl, fontsize=30)
plt.tight_layout()
plt.savefig('pics/%s_mcmc.png' % fstr, dpi=100)

# %%
