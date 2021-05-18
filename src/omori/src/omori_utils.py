#!usr/bin/python3

import numpy as np
import pandas as pd
import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
import random
import datetime


def omori(x, params):
    c, K, p = params
    return K/((x+c)**p)


def omori_syn(c, p, tmin, tmax, N):
    """
    Adapted from tgeobel's GitHub: https://github.com/tgoebel/aftershocks
    Felzer et al. 2002, Triggering of 1999 Mw 7.1 Hector Mine earthquake
    - define create power-law distributed aftershock time vector between tmin and tmax
    - tmin can be = 0, Omori-K parameter is a fct. of all four parameter (tmax-tmin), p, and N
    INPUT:  c, p       - omori parameters describing time shift for complete recording and rate decay exponent
                       - in alphabetical order  
           tmin, tmax  - time window for aftershock catalog
           N           - total number of aftershocks
    """
    vRand = np.random.random_sample( N)
    #===========================================================================
    #          case1:  p != 1
    #===========================================================================    
    #if p != 1.0: #abs(p - 1) < 1e-6:
    p += 1e-4 # this will make it unlikely for p to be exactly 1
    a1 = (tmax + c)**(1-p)
    a2 = (tmin + c)**(1-p)
    a3 = vRand*a1 + (1-vRand)*a2#     
    otimes = a3**(1/(1-p))-c
#     else: # p == 1
#         a1 = np.log( tmax + c)
#         a2 = np.log( tmin + c)
#         a3 = vRand*a1 + (1-vRand)*a2
#         otimes = np.exp( a3) - c
    otimes.sort()
    return otimes


def ogata_logL(otimes, params):
    c, K, p = params
    S, T = min(otimes), max(otimes)
    n = len(otimes)
    if abs(p - 1) < 1e-8:
        A = np.log(T+c) - np.log(S+c)
    else:
        A = ((T+c)**(1-p) - (S+c)**(1-p))/(1-p)
    L = -n*np.log(K) + p*np.sum(np.log(otimes+c)) + K*A
    return L


def bayes_logL(otimes, params):
    c, p = params
    S, T = min(otimes), max(otimes)
    n = len(otimes)
    if abs(p - 1) < 1e-8:
        D = (1/c) / (np.log(1+T/c) - np.log(1+S/c))
    else:
        D = ((1-p)/c) / ((1+T/c)**(1-p) - (1+S/c)**(1-p))
    L = -n*np.log(D) + p*np.sum(np.log(1+otimes/c))
    return L



def bayes_post(otimes, params):
    c, p = params
    S, T = min(otimes), max(otimes)
    n = len(otimes)
    if abs(p - 1) < 1e-8:
        D = (1/c) / (np.log(1+T/c) - np.log(1+S/c))
    else:
        D = ((1-p)/c) / ((1+T/c)**(1-p) - (1+S/c)**(1-p))
    prob = np.prod(D/((1+otimes/c)**p)) * 1/c * 1/p
    return prob



def bayes_getK(otimes, params):
    c, p = params
    S, T = min(otimes), max(otimes)
    n = len(otimes)
    if abs(p - 1) < 1e-8:
        D = (1/c) / (np.log(1+T/c) - np.log(1+S/c))
    else:
        D = ((1-p)/c) / ((1+T/c)**(1-p) - (1+S/c)**(1-p))
    L = -n*np.log(D) + p*np.sum(np.log(1+otimes/c))
    K = D * n * c**p
    return L, K