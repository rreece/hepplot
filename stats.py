"""
hepplot/stats.py

See:
https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
https://en.wikipedia.org/wiki/Chi-square_distribution#Asymptotic_properties
https://www.johndcook.com/blog/wilson_hilferty/
"""


import math
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import scipy.stats


#------------------------------------------------------------------------------
# Basic stats helpers
#------------------------------------------------------------------------------

def poisson_error_up(data):
    y1 = data + 1.0
    d = 1.0 - 1.0/(9.0*y1) + 1.0/(3*math.sqrt(y1))
    return y1*d*d*d-data


def poisson_error_down(data):
    y = data
    if y == 0.0: return 0.0
    d = 1.0 - 1.0/(9.0*y) - 1.0/(3.0*math.sqrt(y))
    return data-y*d*d*d


def clopper_pearson(k, n, cl):
    """
    https://gist.github.com/DavidWalz/8538435
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected
    successes on n trials Clopper Pearson intervals are a conservative
    estimate.
    """
    alpha = 1 - cl
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi


#------------------------------------------------------------------------------
# Pyhf helpers (lower level)
#------------------------------------------------------------------------------

def make_pdf(bkg_data, bkg_uncerts, signal_data):
    """
    TODO:
    -   pyhf.simplemodels.uncorrelated_background
    -   pyhf.simplemodels.correlated_background
    -   manual json
    """
    pdf = pyhf.simplemodels.uncorrelated_background(
            signal=signal_data,
            bkg=bkg_data,
            bkg_uncertainty=bkg_uncerts,
            )
    return pdf


def discovery_p0(pdf,
                 data,
                 init_pars=None,
                 par_bounds=None,
                 fixed_params=None,
                 ):
    if fixed_params is None:
        fixed_params = pdf.config.suggested_fixed()
    q0 = pyhf.infer.test_statistics.q0(
            mu=0.0,
            data=data + pdf.config.auxdata,
            pdf=pdf,
            init_pars=init_pars,
            par_bounds=par_bounds,
            fixed_params=fixed_params,
#            return_fitted_pars=False,
            )
    q0 = float(q0)
    asimov_data = pyhf.infer.calculators.generate_asimov_data(
            asimov_mu=1.0,
            data=data + pdf.config.auxdata,
            pdf=pdf,
            init_pars=init_pars,
            par_bounds=par_bounds,
            fixed_params=fixed_params,
#            return_fitted_pars=False,
            )
    q0A = pyhf.infer.test_statistics.q0(
            mu=0.0,
            data=asimov_data,
            pdf=pdf,
            init_pars=init_pars,
            par_bounds=par_bounds,
            fixed_params=fixed_params,
#            return_fitted_pars=False,
            )
    q0A = float(q0A)
    z0_obs = math.sqrt(q0)
    z0_exp = math.sqrt(q0A)
    z0_exp_band = [z0_exp-2.0, z0_exp-1.0, z0_exp, z0_exp+1.0, z0_exp+2.0]
    q0_exp_band = [math.pow(z, 2) for z in z0_exp_band]
    p0_obs = 1.0 - scipy.stats.norm.cdf(z0_obs)
    p0_exp_band = [1.0-scipy.stats.norm.cdf(z) for z in z0_exp_band]
    return q0, q0_exp_band, z0_obs, z0_exp_band, p0_obs, p0_exp_band


def hypo_test(pdf,
              data,
              mu=1.0,
              init_pars=None,
              par_bounds=None,
              test_stat='qtilde',
              ):
    if init_pars is None:
        init_pars = pdf.config.suggested_init()
    if par_bounds is None:
        par_bounds = pdf.config.suggested_bounds()
    CLs_obs, tail_probs, CLs_exp_band = pyhf.infer.hypotest(mu,
                                                data + pdf.config.auxdata,
                                                pdf,
                                                init_pars,
                                                par_bounds,
                                                return_tail_probs=True,
                                                return_expected_set=True,
                                                test_stat=test_stat,
##### asymptotics
                                                calctype='asymptotics',
#                                                calc_base_dist='clipped_normal', # TODO: check
##### toybased
#                                                calctype='toybased',
#                                                ntoys=2000,
#                                                track_progress=True,
                                                )
    # convert from single element ndarrays to floats
    CLs_obs = float(CLs_obs)
    CLs_exp_band = [float(x) for x in CLs_exp_band]
    # tail_probs (unused)
    CLsb, CLb = tail_probs
    return CLs_obs, CLs_exp_band


def hypo_test_mu_scan(pdf, 
                   data,
                   test_mus=None,
                   init_pars=None,
                   par_bounds=None,
                   mu_bounds=None,
                   mu_step=0.1,
                   test_stat='qtilde',
                   ):
    if init_pars is None:
        init_pars = pdf.config.suggested_init()
    if par_bounds is None:
        par_bounds = pdf.config.suggested_bounds()
    if mu_bounds is None:
        mu_bounds = par_bounds[pdf.config.poi_index]
    else:
        par_bounds[pdf.config.poi_index] = mu_bounds
    if test_mus is None:
        mu_steps = int(round(abs(float(mu_bounds[1] - mu_bounds[0]))/mu_step)) + 1
        test_mus = np.linspace(mu_bounds[0], mu_bounds[1], mu_steps)
    hypo_tests = [
        hypo_test(pdf=pdf,
                  data=data,
                  mu=mu,
                  init_pars=init_pars,
                  par_bounds=par_bounds,
                  test_stat=test_stat,
                  )
        for mu in test_mus
    ]
    # convert hypo_tests from
    # [ (cls_obs1, [cls_exp_band1]), (cls_obs1, [cls_exp_band2]), ...]
    # to
    # cls_obs = [cls_obs1, cls_obs2, ...]
    # cls_exp_band = [[cls_exp_band1], [cls_exp_band2], ...]
    cls_obs = [test[0] for test in hypo_tests]
    cls_exp = [[test[1][i] for i in range(5)] for test in hypo_tests ]
    return cls_obs, cls_exp, test_mus


def invert_interval(cls_obs, cls_exp, test_mus, test_size=0.05):
    """
    Inverts (mu, p-value) by interpolation to the mu value excluded at a CL
    of 1-test_size.
    """
    mu_excl_exp = list()
    n_x = len(cls_obs)
    assert len(cls_exp) == n_x
    for i in range(5):
        _exp = [cls_exp[j][i] for j in range(n_x)]
        mu_excl_exp.append(
            np.interp(
                test_size, list(reversed(_exp)), list(reversed(test_mus))
            )
        )
    mu_excl_obs = np.interp(
        test_size, list(reversed(cls_obs)), list(reversed(test_mus))
    )
    return mu_excl_obs, mu_excl_exp


def twice_nll_mu_scan(pdf, 
                   data,
                   test_mus=None,
                   init_pars=None,
                   par_bounds=None,
                   mu_bounds=None,
                   mu_step=0.1,
                   deltaL=True,
                   ):
    if init_pars is None:
        init_pars = pdf.config.suggested_init()
    if par_bounds is None:
        par_bounds = pdf.config.suggested_bounds()
    params_mle, twice_nll_mle = pyhf.infer.mle.fit(data=data + pdf.config.auxdata,
                                               pdf=pdf, 
                                               init_pars=init_pars, 
                                               par_bounds=par_bounds,
                                               return_fitted_val=True,
                                              )
    mu_mle = params_mle[pdf.config.poi_index]
    
    if mu_bounds is None:
        mu_bounds = par_bounds[pdf.config.poi_index]
    else:
        par_bounds[pdf.config.poi_index] = mu_bounds
    
    if test_mus is None:
        mu_steps = int(round(abs(float(mu_bounds[1] - mu_bounds[0]))/mu_step)) + 1
        test_mus = np.linspace(mu_bounds[0], mu_bounds[1], mu_steps)
        
    twice_nll_results = [
        pyhf.infer.mle.fixed_poi_fit(poi_val=mu,
                                       data=data + pdf.config.auxdata,
                                       pdf=pdf,
                                       init_pars=init_pars,
                                       par_bounds=par_bounds,
                                       return_fitted_val=True,
                                      )
        for mu in test_mus
    ]
    # convert twice_nll_results from
    # [ (params, twice_nll), ...]
    # to twice_nlls
    # [twice_nll, ...]
    twice_nlls = [tnll for _, tnll in twice_nll_results]
    # convert -2 ln L to -2 Delta ln L
    _x = test_mus
    _y = twice_nlls
    min_y = min(_y)
    i_hat = _y.index(min_y)
    x_hat = _x[i_hat]
    if deltaL:
        _y = [ yi-min_y for yi in _y]
        x_hat_m1 = np.interp(
            1.0, list(reversed(_y[:i_hat+1])), list(reversed(_x[:i_hat+1]))
        )
        x_hat_p1 = np.interp(
            1.0, _y[i_hat:], _x[i_hat:]
        )
        x_hat_band = [x_hat_m1, x_hat, x_hat_p1]
        return _y, _x, x_hat_band
    return _y, _x,


#------------------------------------------------------------------------------
# Pyhf helpers (high level)
#------------------------------------------------------------------------------

def hypo_test_signal_scan(
        signal_grid,
        data,
        bkg_data,
        bkg_uncerts,
        mu_bounds=(0, 5),
        mu_step=0.1,
        test_size=0.05,
        ):
    signal_ms = list()
    exclusions_obs = list()
    exclusions_exp = list()
    for signal_point, signal_data in signal_grid.items():
        assert len(signal_point) == 1
        signal_ms.append(signal_point[0])
        pdf = make_pdf(bkg_data=bkg_data,
                       bkg_uncerts=bkg_uncerts,
                       signal_data=signal_data,
                       )
        cls_obs, cls_exp, test_mus = hypo_test_mu_scan(pdf, 
                                                       data, 
                                                       mu_bounds=mu_bounds,
                                                       mu_step=mu_step,
                                                       )
        mu_excl_obs, mu_excl_exp = invert_interval(cls_obs,
                                                   cls_exp,
                                                   test_mus,
                                                   test_size=test_size,
                                                   )
        exclusions_obs.append(mu_excl_obs)
        exclusions_exp.append(mu_excl_exp)
    return signal_ms, exclusions_exp, exclusions_obs


def discovery_p0_signal_scan(
        signal_grid,
        data,
        bkg_data,
        bkg_uncerts,
        ):
    signal_ms = list()
    p0s_obs = list()
    p0s_exp = list()
    for signal_point, signal_data in signal_grid.items():
        assert len(signal_point) == 1
        signal_ms.append(signal_point[0])
        pdf = make_pdf(bkg_data=bkg_data,
                       bkg_uncerts=bkg_uncerts,
                       signal_data=signal_data,
                       )
        p0_results = discovery_p0(pdf, data)
        q0, q0_exp_band, z0_obs, z0_exp_band, p0_obs, p0_exp_band = p0_results
        p0s_obs.append(p0_obs)
        p0s_exp.append(p0_exp_band)
    return signal_ms, p0s_exp, p0s_obs


def signal_strength_signal_scan(
        signal_grid,
        data,
        bkg_data,
        bkg_uncerts,
	    mu_bounds=(-0.5, 2.0),
	    mu_step=0.05,
        ):
	signal_ms = list()
	mu_hat_results = list()
	for signal_point, signal_data in signal_grid.items():
	    assert len(signal_point) == 1
	    signal_ms.append(signal_point[0])
	    pdf = make_pdf(bkg_data=bkg_data,
                       bkg_uncerts=bkg_uncerts,
                       signal_data=signal_data,
                       )
	    twice_nlls, test_mus, mu_hat_band = twice_nll_mu_scan(pdf, 
	                   data,
	                   mu_bounds=mu_bounds,
	                   mu_step=mu_step,
	                   )
	    mu_hat_results.append(mu_hat_band)
	return signal_ms, mu_hat_results


#------------------------------------------------------------------------------
# Stats plotting helpers
#------------------------------------------------------------------------------

def plot_twice_nll(x, y,
           xlabel=None,
           ylabel=None,
           xlim=None,
           ylim=None,
           yline=None,
           yscale=None,
           ):
    fig, ax = plt.subplots()
    ax.plot(x, y, c='black')
    if yline is not None:
        if not isinstance(yline, list):
            ylines = [yline]
        else:
            ylines = yline
        for _yl in ylines:
            ax.plot(x, [_yl] * len(x), color='red', linestyle='dashed')
    if not xlim is False:
        xlim = (x[0], x[-1])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yscale:
        ax.set_yscale('log')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return fig, ax
