#ifndef _BLACK_H_
#define	_BLACK_H_

#include <math.h>


static double std_norm_cdf(double  x){
	return 0.5*(1+erf(x/sqrt(2.0)));
}

static double norm_dist(double  x){
	return (1.0/sqrt(2.0*M_PI))*exp(-0.5*x*x);
}

static double delta(double s, double k, double t,
					double v, double rf, double cp){
	double d1 = (log(s / k) + (0.5 * pow(v, 2)) * t) / (v * sqrt(t));
	return cp * exp(-rf * t) * std_norm_cdf(cp * d1);
}

static double vega(double s, double k, double t,
				   double v, double rf, double cp){
	double d1 = (log(s / k) + (0.5 * pow(v, 2)) * t) / (v * sqrt(t));
	return s * exp(-rf * t) * sqrt(t) * norm_dist(d1);
}

static double tv(double s, double k, double t,
					double v, double rf, double cp){
	double d1 = (log(s / k) + (0.5 * pow(v, 2)) * t) / (v * sqrt(t));
	double d2 = d1 - v * sqrt(t);
	return exp(-rf * t) * (cp * s * std_norm_cdf(cp * d1) - cp * k * std_norm_cdf(cp * d2));
}

#endif