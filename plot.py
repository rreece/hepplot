"""
hepplot/plot.py

See:
https://github.com/CoffeaTeam/coffea/blob/master/coffea/hist/plot.py

TODOs:
[x] Support support arrays as input and convert internal
    calculations to use numpy.
[x]  Make sure everything can be optional: y, yerr, yerrs, data.
[x]  Support symmetric gaussian errors for yerr and yerrs.
[x]  Support only one y in model.
[x]  Add signal histogram lines.
"""


import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

from . import stat


matplotlib.rc('text', usetex = True)
#plt.style.use(hep.style.ATLAS)
plt.style.use([hep.style.CMS, hep.style.firamath])


def hist1d(bins,
           y=None,
           yerr=None,
           yerrs=None,
           labels=None,
           data=None,
           data_err=None,
           data_label=None,
           signals=None,
           signal_labels=None,
           xlabel=None,
           ylabel=None,
           unit=None,
           ratio=False,
           ratio_label=None,
           stack_signals=True,
           ):

    assert not ((yerr is not None) and (yerrs is not None))

    n_bins = len(bins)-1
    n_samples = 0

    if y is not None:
        if isinstance(y, list) and isinstance(y[0], (list, np.ndarray)):
            n_samples = len(y)
            for y_i in y:
                assert n_bins == len(y_i)
            y = y.copy()
            y.reverse()
            if labels is not None:
                labels = labels.copy()
                labels.reverse()
        else:
            assert n_bins == len(y), "%i, %i" % (n_bins, len(y))
            n_samples = 1
            # if y one sample, and not a list of samples, make it one
            y = [y]
            if not isinstance(labels, list):
                labels = [labels]

    ## set default poisson errors for data
    if (data is not None) and (data_err is None):
        data_err = np.zeros((n_bins, 2), dtype=np.float32)
        for j_bin in range(n_bins):
            data_err[j_bin][0] = stat.poisson_error_up(data[j_bin])
            data_err[j_bin][1] = stat.poisson_error_down(data[j_bin])

    ## convert to np.arrays if needed
    bins = np.asarray(bins, dtype=np.float32)
    if y is not None:
        if isinstance(y, list):
            y = [ np.asarray(y_i, dtype=np.float32) for y_i in y ]
        else:
            y = np.asarray(y, dtype=np.float32)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=np.float32)
    if yerrs is not None:
        assert isinstance(yerrs, list)
        yerrs = [ np.asarray(yerr_i, dtype=np.float32) for yerr_i in yerrs ]
    if data is not None:
        data = np.asarray(data, dtype=np.float32)
    if signals is not None:
        assert isinstance(signals, list)
        signals = [ np.asarray(signal_i, dtype=np.float32) for signal_i in signals ]

    ## allow gaussian symmetric yerr
    if yerr is not None:
        if isinstance(yerr[0], (list, np.ndarray)):
            assert len(yerr[0]) == 2
        else:
            _yerr = np.zeros((n_bins, 2), dtype=np.float32)
            for j_bin in range(n_bins):
                _yerr[j_bin][0] = yerr[j_bin]
                _yerr[j_bin][1] = yerr[j_bin]
            yerr = _yerr

    ## allow gaussian symmetric yerrs
    if yerrs is not None:
        if isinstance(yerrs[0][0], (list, np.ndarray)):
            assert len(yerrs[0][0]) == 2
        else:
            _yerrs = [ np.zeros((n_bins, 2), dtype=np.float32) for _ in yerrs ]
            for i_sample in range(n_samples):
                for j_bin in range(n_bins):
                    _yerrs[i_sample][j_bin][0] = yerrs[i_sample][j_bin]
                    _yerrs[i_sample][j_bin][1] = yerrs[i_sample][j_bin]
            yerrs = _yerrs

    ## allow gaussian symmetric data_err
    if data_err is not None:
        if isinstance(data_err[0], (list, np.ndarray)):
            assert len(data_err[0]) == 2
        else:
            _data_err = np.zeros((n_bins, 2), dtype=np.float32)
            for j_bin in range(n_bins):
                _data_err[j_bin][0] = data_err[j_bin]
                _data_err[j_bin][1] = data_err[j_bin]
            data_err = _data_err

    ## convert yerr from [bin][up,down] to [down,up][bin]
    if yerr is not None:
        yerr = np.swapaxes(yerr, 0, 1)
        yerr = np.flip(yerr, 0)

    ## convert yerrs from [sample][bin][up,down] to [sample][down,up][bin]
    if yerrs is not None:
        for _i in range(n_samples):
            yerrs[_i] = np.swapaxes(yerrs[_i], 0, 1)
            yerrs[_i] = np.flip(yerrs[_i], 0)

    ## convert data_err from [bin][up,down] to [down,up][bin]
    if data_err is not None:
        data_err = np.swapaxes(data_err, 0, 1)
        data_err = np.flip(data_err, 0)

    ## calculate ytotal
    ytotal = None
    if y is not None:
        ytotal  = np.zeros((n_bins,), dtype=np.float32)
        for _i in range(n_samples):
            ytotal  = ytotal + y[_i]

    ## make top subplot
    fig = plt.figure()
    if (data is not None) and ratio:
        gs = fig.add_gridspec(2, 1, height_ratios=(3, 1),
                              wspace=0, hspace=0.04)
        ax1 = fig.add_subplot(gs[0, 0])
    else:
        fig, ax1 = plt.subplots()
    axes = list()
    axes.append(ax1)

    bincenters = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)
    binwidths = np.asarray([bins[i+1]-bins[i] for i in range(n_bins)], dtype=np.float32)

    ## plot hist stack
    if y is not None:
        # TODO: make colors configurable
        if n_samples > 1:
            colors = [plt.cm.Spectral(i/float(n_samples-1)) for i in range(n_samples)]
        else:
            colors = ['lightgray']
        weights = y
        binned = [np.asarray(bins[:-1], dtype=np.float32) for _ in range(n_samples)]

        plt.hist(binned, bins,
            weights=weights,
            stacked=True,
            density=False,
            color=colors,
            label=labels,
            )

    # TODO
    signal_colors = ['orange', 'cyan', 'lime', 'magenta']

    ## plot signals
    n_signals = 0
    if signals is not None:
        n_signals = len(signals)
        if stack_signals: # stack signals on top of ytotal
            for i_sig in range(n_signals):
                for j_bin in range(n_bins):
                    signals[i_sig][j_bin] = signals[i_sig][j_bin] + ytotal[j_bin]
        binned = [np.asarray(bins[:-1], dtype=np.float32) for _ in range(n_signals)]
        plt.hist(binned, bins,
            weights=signals,
            stacked=False,
            density=False,
            label=signal_labels[:n_signals],
            color=signal_colors[:n_signals],
            histtype='step',
            linewidth=2,
            fill=False,
            )

    ## sum yerrs to yerr
    if yerrs is not None:
        yerr = np.zeros((2, n_bins), dtype=np.float32)

        for _i in range(n_samples):
            yerr = yerr + (yerrs[_i]*yerrs[_i])
        yerr = np.sqrt(yerr)

    ## plot error band
    if yerr is not None:
        xerr = [
                [w/2 for w in binwidths],
                [w/2 for w in binwidths],
            ]
        xerr = np.asarray(xerr)
        uncert_boxes = make_error_boxes(ax1, bincenters, ytotal, xerr, yerr,
                                        hatch='///')

    ## plot data
    if data is not None:
        plt.errorbar(bincenters, data, yerr=data_err, 
            label=data_label,
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=2,
            capsize=0,
            markersize=8,
            zorder=100,
            )
    
    ## axis labels
    _label = ''
    if ylabel:
        _label = ylabel
        if unit:
            _label += (' / (%g %s)' % (binwidths[0], unit))
        plt.ylabel(_label)
    if not ratio:
        _label = ''
        if xlabel:
            _label = xlabel
            if unit:
                _label += ' [%s]' % (unit)
            plt.xlabel(_label)

    ## make legend
    if labels or data_label:
        leg_handles, leg_labels = ax1.get_legend_handles_labels()
    
        if (data is not None) and data_label:
            data_handle = leg_handles.pop()
            assert leg_labels.pop() == data_label

        if (signals is not None) and signal_labels:
            signal_handels = list()
            for _ in range(n_signals):
                signal_handels.append(leg_handles.pop())
                assert leg_labels.pop() in signal_labels

        if labels:
            leg_handles.reverse()
            leg_labels.reverse()

        if yerr is not None:
            leg_handles.append(Patch(facecolor='darkgray',
                                     alpha=0.4,
                                     hatch='///'))
            leg_labels.append('Uncert.')

        if (data is not None) and data_label:
            leg_handles.append(data_handle)
            leg_labels.append(data_label)

        if (signals is not None) and signal_labels:
            leg_handles.extend(signal_handels)
            leg_labels.extend(signal_labels)

        total_mean = 0.
        if ytotal is not None:
            sum_ytotal = sum(ytotal)
            total_mean = sum([y_i*x_i/sum_ytotal for y_i, x_i in zip(ytotal, bincenters)])
        elif data is not None:
            sum_data = sum(data)
            total_mean = sum([y_i*x_i/sum_data for y_i, x_i in zip(data, bincenters)])

        middle_of_range = (bins[-1] - bins[0])/2

        leg_loc = 'upper left'
        if total_mean < middle_of_range:
            leg_loc = 'upper right'

        leg = plt.legend(leg_handles, leg_labels, loc=leg_loc)
    
    ## make ratio
    if (data is not None) and ratio:
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        axes.append(ax2)
        
        plt.axhline(y=1.0, color='darkgray', linestyle='-', zorder=-1)
        
        y_ratio = [d_i/y_i if y_i else 0. for d_i, y_i in zip(data, ytotal)]
        y_ratio_err = [
            [de_i/d_i for d_i, de_i in zip(data, data_err[0])],
            [de_i/d_i for d_i, de_i in zip(data, data_err[1])],
        ]

        ## plot error band
        if yerr is not None:
            y_ratio_band = [
                [ye_i/y_i for y_i, ye_i in zip(ytotal, yerr[0])],
                [ye_i/y_i for y_i, ye_i in zip(ytotal, yerr[1])],
            ]
        
            xerr = [
                    [w/2 for w in binwidths],
                    [w/2 for w in binwidths],
                ]
            make_error_boxes(ax2, bincenters, [1.0]*n_bins, xerr, y_ratio_band)
#                             hatch='///')
        
        ## plot ratio
        plt.errorbar(bincenters, y_ratio, yerr=y_ratio_err,
            label='ratio',
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=2,
            capsize=0,
            markersize=8,
            zorder=100,
            )
        
        ## axis labels
        if ratio_label:
            ax2.set_ylabel(ratio_label)
        #ax2.set_ylim(0.7, 1.3) # HACK
        _label = ''
        if xlabel:
            _label = xlabel
            if unit:
                _label += ' [%s]' % (unit)
            plt.xlabel(_label)
    
        fig.subplots_adjust(wspace=0, hspace=0)

    return fig, axes


def make_error_boxes(ax, xdata, ydata, xerror, yerror,
                     facecolor='darkgray',
#                     edgecolor='none',
                     alpha=0.4,
                     hatch=None,
                     zorder=20):
    """
    From:
    https://matplotlib.org/3.1.0/gallery/statistics/errorbars_and_boxes.html
    """

    # Create list for all the error patches
    errorboxes = []

    xerror = np.asarray(xerror)
    yerror = np.asarray(yerror)

    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes,
                         facecolor=facecolor,
                         alpha=alpha,
#                         edgecolor=edgecolor,
                         hatch=hatch,
                         zorder=zorder)

    # Add collection to axes
    ax.add_collection(pc)

    return pc


def plot_brazil(x, exp, obs,
                xlabel=None,
                ylabel=None,
                xlim=None,
                ylim=None,
                yline=None,
                ):
    """
    Plot a series of hypothesis tests for various POI values.

    def plot_results(ax, mutests, tests, test_size=0.05):
        cls_obs = np.array([test[0] for test in tests]).flatten()
        cls_exp = [np.array([test[1][i] for test in tests]).flatten() for i in range(5)]
    """
    fig, ax = plt.subplots()
    for idx, color in zip(range(5), 5 * ['black']):
        ax.plot(
            x, exp[idx], c=color, linestyle='dotted' if idx != 2 else 'dashed'
        )
    ax.fill_between(x, exp[0], exp[-1], facecolor='yellow')
    ax.fill_between(x, exp[1], exp[-2], facecolor='green')
    ax.plot(x, obs, c='black')
    if yline is not None:
        ax.plot(x, [yline] * len(x), c='red')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return fig, ax
