#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:13:06 2021

@author: mgm
"""
#%% imports 
import numpy as np
import matplotlib.pyplot as plt
import batman
import emcee
import corner

import os
os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import cpu_count
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))


#%%
#initialize_data
data = np.loadtxt('/Users/mgm/Desktop/example_data.txt')
time = data[:,0]
folded_flux = data[:,1]
folded_err = data[:,2]
flux = folded_flux
err = folded_err
t = time*8.376608

plt.figure(figsize=(12,16))
plt.errorbar(time,flux,err,ls='',marker='.',color='k',alpha=0.6)
plt.xlabel('Orbital Phase',size=16)
plt.ylabel('Normalized Flux',size=16)
plt.xticks(fontsize=16)
plt.xticks(fontsize=16)
plt.show() 

#%% define functions aand perform fitting
def logprob(p0,params,t,flux,err,prior_lims,prior_types):
    params.rp = p0[0]
    params.a = p0[1]
    params.inc = p0[2]
    params.u = [p0[3],p0[4]]
    m = batman.TransitModel(params,t)    #initializes model
    batflux = m.light_curve(params)
    negLnLikelihood = -0.5*np.nansum((((flux-batflux)**2)/err**2) + (np.log(2*np.pi*err**2)))
    lp = log_prior(p0,prior_lims,prior_types)
    negLnLikelihood = negLnLikelihood + lp 
    return negLnLikelihood

def log_prior(p0,prior_lims,prior_types):
    log_prior = 0.0
    for i, prior_type in enumerate(prior_types):
        if prior_type == 'gaussian':
            log_prior += gaussian_log_prior(p0[i],prior_lims[i][0],prior_lims[i][1])
        elif prior_type == 'linear':
            log_prior += uniform_log_prior(p0[i],prior_lims[i][0],prior_lims[i][1])		
    return log_prior

def uniform_log_prior(param, lower_bound, upper_bound):
	norm_constant = 1/(upper_bound - lower_bound)
	if lower_bound < param < upper_bound:
		return np.log(norm_constant)
	return -np.inf

def gaussian_log_prior(param, prior, prior_sigma):
	return -0.5*(np.log(2*np.pi*prior_sigma**2) + \
		 ((param-prior)/prior_sigma)**2)

def log_prob_data_global(p0,params,prior_lims,prior_types):
    params.rp = p0[0]
    params.a = p0[1]
    params.inc = p0[2]
    params.u = [p0[3],p0[4]]
    m = batman.TransitModel(params,t)    #initializes model
    batflux = m.light_curve(params)
    negLnLikelihood = -0.5*np.nansum((((folded_flux-batflux)**2)/folded_err**2) + (np.log(2*np.pi*folded_err**2)))
    lp = log_prior(p0,prior_lims,prior_types)
    negLnLikelihood = negLnLikelihood + lp 
    return negLnLikelihood   
        
def fit_transit(planet_params,prior_lims,prior_types):     
    per,rp,inc,a,u = planet_params[0],planet_params[1],planet_params[2],planet_params[3],planet_params[4]
    phase, folded_flux, folded_err = time,flux,err
    
    #initialize the model and intial guess for MCMC
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = per/2                     #time of inferior conjunction
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    params.per = per                      #orbital period
    t = phase*per                         #time for fitting 

    p0_guess = np.array([rp,a,inc,u[0],u[1]])
    
    #run the MCMC
    ndim, nwalkers = len(p0_guess), 50
    pos = p0_guess + p0_guess*1e-7*np.random.randn(50, len(p0_guess))
    nwalkers, ndim = pos.shape
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob,args=(params,t,folded_flux,folded_err,prior_lims,prior_types))
    #state = sampler.run_mcmc(pos, 500, progress=True)
    #sampler.reset()
    #sampler.run_mcmc(state, 3000, progress=True)  
    #####use parallelization to perform the MCMC
    from multiprocessing import Pool
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_data_global,args=(params,prior_lims,prior_types),moves=emcee.moves.DEMove(),pool=pool)
        state = sampler.run_mcmc(pos, 500, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, 3000, progress=True)
  
    #check acceptance fractions and create corner plot
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    #print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time())))
    flat_samples = sampler.get_chain(discard=0, flat=True) #thin = half the autocorrelation time
    labelssss = ["${R_p}/{R_*}$", "$a/{R_*}$","inc","$u_1$","$u_2$"]
    fig = corner.corner(flat_samples, labels=labelssss, quantiles=[0.160, 0.500, 0.840],levels=[1 - np.exp(-(1**2)/2),1 - np.exp(-(2**2)/2),1 - np.exp(-(3**2)/2)],
                           show_titles=True, title_kwargs={"fontsize": 14},title_fmt='.4f',label_kwargs=dict(fontsize=14))
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=9)
        
    #draw the median of the chain as the best fit
    p0_best = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        p0_best.append(mcmc[1])

    #define a function to generate model points
    def batflux(p0_best):
        params.rp = p0_best[0]                    #planet radius (in units of stellar radii)
        params.a = p0_best[1]                        #semi-major axis (in units of stellar radii)
        params.inc = p0_best[2]                      #orbital inclination (in degrees)
        params.u = [p0_best[3],p0_best[4]]
        m = batman.TransitModel(params,t)    #initializes model
        batflux = m.light_curve(params)
        return batflux
    
    #bin the data for a finer model comparison
    start,end = phase[0],phase[len(phase)-1]
    binsize = ((5/1440)/per)
    nbins = int(np.floor((end-start)/binsize))
    binnedphase, binnedflux, binnederrs = [],[],[]
    for i in range(nbins):
        binstart = start+(i*binsize)
        binend = start+((i+1)*binsize)
        bindexes = np.where(np.logical_and(phase>binstart,phase<binend))
        avgphase = np.nanmedian(phase[bindexes])
        avgflux = np.nanmedian(folded_flux[bindexes])
        binnedphase.append(avgphase)
        binnedflux.append(avgflux)
        binnederrs.append(np.nanstd(folded_flux[bindexes])/np.sqrt(len(folded_flux[bindexes])))
    
    #plot the data, the best fit model, and some random posterior draws 
    plt.figure(figsize=(8,8))
    plt.errorbar(phase, folded_flux, folded_err,ls='',marker='.',color='gray',alpha=0.3,zorder=1)
    plt.errorbar(binnedphase, binnedflux, binnederrs,ls='',marker='.',color='k',alpha=1.0,label='5 min binned',zorder=5)
    plt.plot(phase,batflux(p0_best),'r.',label='model',zorder=10)
    for theta in flat_samples[np.random.randint(len(flat_samples), size=99)]:
        plt.plot(phase,batflux(theta),'r-',alpha=0.1,zorder=1)
    plt.legend()
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    plt.show()

#####defining constants
Rearth = 6371000
Rsun = 6.957e8
Msun = 1.989e30
G = 6.674e-11
####defining system parameters 
Rstar = 0.900438*Rsun
Mstar = 0.932*Msun
rp_rs = (9.877717*Rearth)/Rstar
period = 8.376608
t0 = 2459202.35532-2457000
duration = 2.171548/24
a_rs =  (((((period*86400)**2)*G*Mstar)/(4*(np.pi**2)))**(1/3))/Rstar
####setting up MCMC inputs and running
prior_lims = [[0.0,0.2],[a_rs,a_rs*0.25],[85,90],[0.0,1.0],[0.0,1.0]]
prior_types = ['linear','gaussian','linear','linear','linear']
planet_params = [period,rp_rs,87.4,a_rs,[0.2552,0.3873]]
fit_transit(planet_params,prior_lims,prior_types)

