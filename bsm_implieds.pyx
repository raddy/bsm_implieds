import numpy as np
cimport numpy as np
import pandas as pd
from libc.math cimport exp, sqrt, pow, log, erf, abs, M_PI
cimport cython

ctypedef np.double_t DTYPE_t


cdef extern from "black.h":
    double tv(double s, double k, double t,double v, double rf, double cp)
    double vega(double s, double k, double t,double v, double rf, double cp)
    double delta(double s, double k, double t,double v, double rf, double cp)

def bs_tv(double s, double k, double t,double v, double rf, double cp):
    return tv(s,k,t,v,rf,cp)

@cython.cdivision(True)
@cython.boundscheck(False)
def implied_vol(double underlying, double price, double strike, double t, double rf, double cp):
    cdef long i = 0
    cdef double prices_guess, vol_guess = 1
    cdef double diff, delt
    
    for i in range(0,20):
        price_guess = tv(underlying,strike,t,vol_guess,rf,cp)
        diff = price - price_guess
        if abs(diff) < .001:
            return vol_guess
        vegalol = vega(underlying,strike,t,vol_guess,rf,cp)
        if vegalol<.01:
            return -1
        vol_guess += diff / vegalol
    return -1

@cython.cdivision(True)
@cython.boundscheck(False)
def implied_fut(double guess, double price, double strike, double t, double rf, double sigma, double cp):
    cdef long i
    cdef double prices_guess, underlying_guess = guess
    cdef double diff, delt
    
    if price <= .01:
        return np.NaN
    
    for i in range(20):
        price_guess = tv(underlying_guess,strike,t,sigma,rf,cp)
        diff = price - price_guess
        if abs(diff) < .0001:
            return long(underlying_guess*10000) / 10000.0
        delt = delta(underlying_guess,strike,t,sigma,rf,cp)
        underlying_guess += diff / delt
    return np.NaN


#quick functions to go back and from call tvs to vols and vv.
def vols_to_tvs(vs,ks,spot,tte,ir=.03,cp=1):
    res = []
    for v,k in zip(vs,ks):
        res.append(bs_tv(spot,k,tte,v,ir,cp))
    return res

def tvs_to_vols(tvs,ks,spot,tte,ir=.03,cp=1):
    res = []
    for tv,k in zip(tvs,ks):
        res.append(implied_vol(spot,tv,k,tte,ir,cp))
    return res

def vol_cost_function(predicted_vols,observed_vols,bid_ask_tick_widths,market_vegas):
    weights = 1./bid_ask_tick_widths * sqrt(market_vegas)
    step1 =  pd.Series(weights * (predicted_vols-observed_vols)).dropna() 
    return step1.abs().sum()

@cython.cdivision(True)
@cython.boundscheck(False)
def implied_futs(np.ndarray[DTYPE_t, ndim=1] prices,np.ndarray[DTYPE_t, ndim=1] strikes,
    np.ndarray[DTYPE_t, ndim=1] vols, np.ndarray[DTYPE_t, ndim=1] ttes, np.ndarray[long, ndim=1] types,
    double rf,double guess):
    
    cdef long price_len = prices.shape[0], strike_len = strikes.shape[0], vols_len = vols.shape[0], tte_len = ttes.shape[0],type_len = types.shape[0],i=0
    assert(price_len==strike_len)
    assert(strike_len==vols_len)
    assert(vols_len==tte_len)
    assert(tte_len==type_len)

    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(price_len, dtype=np.double) * np.NaN


    for i in range(0,price_len):
        if (prices[i]>=0) and  (vols[i]>0) and types[i]!=0:
            res[i] = implied_fut(guess,prices[i],strikes[i],ttes[i],rf,vols[i],types[i])
        elif prices[i]>=0 and types[i]==0:
            res[i] = prices[i]
        else:
            res[i] = np.NaN
    return res    
