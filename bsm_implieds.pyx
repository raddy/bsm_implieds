import numpy as np
cimport numpy as np
import pandas as pd
from libc.math cimport exp, sqrt, pow, log, erf, abs, M_PI
import heapq
cimport cython

ctypedef np.double_t DTYPE_t


cdef extern from "black.h":
    double tv(double s, double k, double t,double v, double rf, double cp)
    double vega(double s, double k, double t,double v, double rf, double cp)
    double delta(double s, double k, double t,double v, double rf, double cp)

#any trade that happens above the previous bid is a 'buy'
cdef inline int trade_dir(double prev_bid,double trade_price):
    if trade_price>prev_bid:
        return 1
    return -1


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
def deltas(np.ndarray[DTYPE_t, ndim=1] underlyings,
    np.ndarray[DTYPE_t, ndim=1] strikes,
    np.ndarray[DTYPE_t, ndim=1] vols, np.ndarray[DTYPE_t, ndim=1] ttes, np.ndarray[long, ndim=1] types,
    double rf):
    
    cdef:
        long und_len = underlyings.shape[0]
        long strike_len = strikes.shape[0]
        long vols_len = vols.shape[0]
        long tte_len = ttes.shape[0]
        long type_len = types.shape[0]
        long i=0
        double last_underlying,approx_delta
    assert(und_len == strike_len)
    assert(strike_len==vols_len)
    assert(vols_len==tte_len)
    assert(tte_len==type_len)

    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(und_len, dtype=np.double) * np.NaN

    for i in range(0,und_len):
        last_underlying = underlyings[i]
        if last_underlying>0:
            res[i] = delta(last_underlying,strikes[i],ttes[i],vols[i],rf,types[i])
        else:
            res[i] = np.NaN
    return res

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

def net_effect_cython(np.ndarray[object, ndim=1] syms, 
        np.ndarray[double, ndim=1] bids,
        np.ndarray[double, ndim=1] trade_prices,
        np.ndarray[long, ndim=1] trade_sizes):
    cdef:
        long bids_len = bids.shape[0]
        long syms_len = syms.shape[0]
        long trade_prices_len = trade_prices.shape[0]
        long trade_sizes_len = trade_sizes.shape[0]
        long i=0,j
        dict last_bid = {}
        np.ndarray[DTYPE_t, ndim=1] effect = np.zeros(syms_len, dtype=np.double) * np.NaN
        object sym

    #lol asserts mess
    assert(syms_len==bids_len)
    assert(syms_len==trade_prices_len)
    assert(trade_prices_len==trade_sizes_len)


    for i from 0 <= i < syms_len:
        sym = syms[i]
        if trade_sizes[i]>0: #sup there's a trade
            if last_bid.has_key(sym):
                effect[i] = trade_dir(last_bid[sym],trade_prices[i]) * trade_sizes[i]
            else:
                effect[i] = trade_sizes[i]
        last_bid[sym] = bids[i]
    return effect

cdef double approximate_abs_delta(long opt_type, double underlying,double strike,double tte, double vol, double risk_free):
    if opt_type==0: ##future
        return 1
    return abs(delta(underlying,strike,tte,vol,risk_free,opt_type))

cdef int no_md(double bid1,double ask1):
    return not (bid1>0 and ask1>0)

cdef int throwout(long opt_type,double approx_delta):
    if opt_type==0:
        return 0
    return approx_delta < .2

cdef int no_underlying(double und):
    return not und>0

def implied_info(long opt_type,double bid1,double bid1s,double bid2,double bid2s,double ask1,double ask1s,double ask2,double ask2s,double synthetic,
        double basis,double vol,double tte,double strike,double approx_delta,double rf,double guess):
    res = np.zeros(8)
    if opt_type == 0:
        return np.array([bid1,bid1s,bid2,bid2s,ask1,ask1s,ask2,ask2s])
    
    
    
    if opt_type == 1:
        res[1] = bid1s*approx_delta
        res[3] = bid2s*approx_delta
        res[5] = ask1s*approx_delta
        res[7] = ask2s*approx_delta
        
        if res[1]<1:
            res[0] = 0
        else:
            res[0] = implied_fut(synthetic,bid1,strike,tte,rf,vol,opt_type) - basis
        if res[3]<1:
            res[2]  = 0
        else:
            res[2] = implied_fut(synthetic,bid2,strike,tte,rf,vol,opt_type) - basis
        if res[5]<1:
            res[4] = 999999
        else:
            res[4] = implied_fut(synthetic,ask1,strike,tte,rf,vol,opt_type) - basis
        if res[7]<1:
            res[6] = 999999
        else:
            res[6] = implied_fut(synthetic,ask2,strike,tte,rf,vol,opt_type) - basis
        
    elif opt_type == -1:
        res[1] = ask1s*approx_delta
        res[3] = ask2s*approx_delta
        res[5] = bid1s*approx_delta
        res[7] = bid2s*approx_delta
        
        if res[1]<1:
            res[0] = 0
        else:
            res[0] = implied_fut(synthetic,ask1,strike,tte,rf,vol,opt_type) - basis
        if res[3]<1:
            res[2]  = 0
        else:
            res[2] = implied_fut(synthetic,ask2,strike,tte,rf,vol,opt_type) - basis
        if res[5]<1:
            res[4] = 999999
        else:
            res[4] = implied_fut(synthetic,bid1,strike,tte,rf,vol,opt_type) - basis
        if res[7]<1:
            res[6] = 999999
        else:
            res[6] = implied_fut(synthetic,bid2,strike,tte,rf,vol,opt_type) - basis
        
        
    
    return res

class TopN(object):
    """
    v format: (num, value)

    after looking into http://hg.python.org/cpython/file/2.7/Lib/heapq.py, 
    i find heappushpop already optimize, no need bottom value

    feed() can be optimize further, if needed:
        using func object instead of compare len(self.h) each time
    """
    def __init__(self, N):
        self.N = N
        self.h = []        

    def feed(self, v):  
        if len(self.h) < self.N:
            heapq.heappush(self.h, v)
        else:
            heapq.heappushpop(self.h, v)

    def result(self):
        self.h.sort(reverse=True)
        return self.h

def cy_parse_dict(some_res_dict):
    keys = some_res_dict.keys()
    topbids = TopN(5)
    topasks = TopN(5)
    for k in keys:
        stuff = some_res_dict[k]
        if stuff[0]>0:
            topbids.feed((stuff[0], (k,stuff[1])))
        if stuff[2]>0:
            topbids.feed((stuff[2], (k,stuff[1])))
        if stuff[4]>0:
            topasks.feed((stuff[4]*-1, (k,stuff[1])))
        if stuff[6]>0:
            topasks.feed((stuff[6]*-1, (k,stuff[1])))
    return [topbids.result(),topasks.result()]  

#assumes you pass in ONLY KOPSI message 
#works on ONLY 2 level books
def fast_implieds(np.ndarray[object, ndim=1] syms, np.ndarray[DTYPE_t, ndim=1] bid1,np.ndarray[long, ndim=1] bid1s,np.ndarray[DTYPE_t, ndim=1] bid2,
        np.ndarray[long, ndim=1] bid2s,np.ndarray[DTYPE_t, ndim=1] ask1,np.ndarray[long, ndim=1] ask1s,np.ndarray[DTYPE_t, ndim=1] ask2,
        np.ndarray[long, ndim=1] ask2s,np.ndarray[DTYPE_t, ndim=1] synthetics,np.ndarray[DTYPE_t, ndim=1] basis,np.ndarray[DTYPE_t, ndim=1] vols,
        np.ndarray[DTYPE_t, ndim=1] ttes,np.ndarray[DTYPE_t, ndim=1] strikes,np.ndarray[long, ndim=1] types,double rf, double guess):
    cdef:
        long sym_len = len(syms)
        double approx_delta
        object sym
        dict last_info = {}
        dict last_md = {}
        long throwout_count = 0,null_count=0, redundant_count = 0
        np.ndarray[DTYPE_t, ndim=2] top5bids = np.zeros([sym_len,10],dtype=np.double) * np.NaN
        np.ndarray[object,ndim=2] top5bid_symbols = np.zeros([sym_len,5],dtype=object) * np.NaN
        np.ndarray[DTYPE_t, ndim=2] top5asks = np.zeros([sym_len,10],dtype=np.double) * np.NaN
        np.ndarray[object,ndim=2] top5ask_symbols = np.zeros([sym_len,5],dtype=object) * np.NaN
        
    for i from 0 <= i < sym_len:
        sym = syms[i]
        if no_md(bid1[i],ask1[i]) or (types[i]!=0 and no_underlying(synthetics[i])):
            last_info[sym] = [np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]
            null_count+=1
            continue

        approx_delta = approximate_abs_delta(types[i],synthetics[i],strikes[i],ttes[i],vols[i],rf)
        if throwout(types[i],approx_delta):
            last_info[sym] = [np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]
            throwout_count+=1
            continue
        if last_md.has_key(sym) and np.allclose(last_md[sym],[bid1[i],bid1s[i],ask1[i],ask1s[i]]):
            redundant_count+=1
            continue
        last_info[sym] = implied_info(types[i],bid1[i],bid1s[i],bid2[i],bid2s[i],ask1[i],ask1s[i],ask2[i],ask2s[i],synthetics[i],
            basis[i],vols[i],ttes[i],strikes[i],approx_delta,rf,guess)
        last_md[sym] = [bid1[i],bid1s[i],ask1[i],ask1s[i]]
        keys = last_info.keys()
        if len(keys)>0:
            bbids,basks = cy_parse_dict(last_info)
            for j from 0 <= j < len(bbids):
                top5bid_symbols[i][j] = bbids[j][1][0]
                top5bids[i][j*2] = bbids[j][0]
                top5bids[i][j*2+1] = bbids[j][1][1]
            for j from 0 <= j < len(basks):
                top5ask_symbols[i][j] = basks[j][1][0]
                top5asks[i][j*2] = basks[j][0]*-1
                top5asks[i][j*2+1] = basks[j][1][1]
    print 'Encountered %d null values and threw out due to delta limits %d values encountering %d redundant pieces' % (null_count,throwout_count,redundant_count)
    top5bids_df = pd.DataFrame(top5bids,columns=['BidPrice1','BidSize1','BidPrice2','BidSize2','BidPrice3','BidSize3','BidPrice4','BidSize4','BidPrice5','BidSize5'])
    top5asks_df = pd.DataFrame(top5asks,columns=['AskPrice1','AskSize1','AskPrice2','AskSize2','AskPrice3','AskSize3','AskPrice4','AskSize4','AskPrice5','AskSize5'])
    top5bid_symbols_df = pd.DataFrame(top5bid_symbols,columns=['BidSymbol1','BidSymbol2','BidSymbol3','BidSymbol4','BidSymbol5'])
    top5ask_symbols_df = pd.DataFrame(top5ask_symbols,columns=['AskSymbol1','AskSymbol2','AskSymbol3','AskSymbol4','AskSymbol5'])
    df =  top5bid_symbols_df.join(top5bids_df).join(top5ask_symbols_df).join(top5asks_df)
    return df.fillna(method='ffill')
