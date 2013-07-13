import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, log, erf, abs, M_PI
cimport cython

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