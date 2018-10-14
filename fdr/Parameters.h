#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

struct Parameters {
    double lambda;
    double gamma;
    double minimum_inlier_rate;
    double eta_zero;
    double seg_k;
    int tau;
    int bins_width;
    bool use_swap;

    Parameters(double lambda = 0.3, double gamma = 20,
               double minimum_inlier_rate = 0.50, double eta_zero = 0.001,
               double seg_k = 30, int tau = 16, int bins_width = 2,
               bool use_swap = true)

        : lambda(lambda),
          gamma(gamma),
          minimum_inlier_rate(minimum_inlier_rate),
          eta_zero(eta_zero),
          seg_k(seg_k),
          tau(tau),
          bins_width(bins_width),
          use_swap(use_swap) {}
};

#endif //_PARAMETERS_H_
