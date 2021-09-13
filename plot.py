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

from . import stats


#matplotlib.rc('text', usetex = True)
#plt.style.use(hep.style.ATLAS)
plt.style.use([hep.style.CMS, hep.style.firamath])
plt.style.use([{
    'xaxis.labellocation': 'center',
    'yaxis.labellocation': 'center',
    }])


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
           signal_colors=None,
           xlabel=None,
           ylabel=None,
           ycolors=None,
           yalpha=None,
           xlabels=None,
           stacked=True,
           xminorticks=True,
           unit=None,
           ratio=False,
           ratio_label=None,
           stack_signals=True,
           xlim=None,
           ylim=None,
           yscale=None,
           data_color='black',
           figsize=None,
           fontsize=None,
           xlabels_fontsize=None,
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
            if ycolors is not None:
                ycolors.reverse()
        else:
            assert n_bins == len(y), "%i, %i" % (n_bins, len(y))
            n_samples = 1
            # if y one sample, and not a list of samples, make it one
            y = [y]
            if (labels is not None) and (not isinstance(labels, list)):
                labels = [labels]

    ## set default poisson errors for data
    if (data is not None) and (data_err is None):
        data_err = np.zeros((n_bins, 2), dtype=np.float32)
        for j_bin in range(n_bins):
            data_err[j_bin][0] = stats.poisson_error_up(data[j_bin])
            data_err[j_bin][1] = stats.poisson_error_down(data[j_bin])

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
        if isinstance(data_err[0], (tuple, list, np.ndarray)):
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
        fig, ax1 = plt.subplots(figsize=figsize)

    if yscale:
        ax1.set_yscale('log')

    axes = list()
    axes.append(ax1)

    bincenters = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)
    binwidths = np.asarray([bins[i+1]-bins[i] for i in range(n_bins)], dtype=np.float32)

    ## plot hist stack
    if y is not None:
        # TODO: make colors configurable
        if ycolors is not None:
            colors=ycolors
        elif n_samples > 1:
            colors = [plt.cm.Spectral(i/float(n_samples-1)) for i in range(n_samples)]
        else:
            colors = ['lightgray']
        weights = y
        binned = [np.asarray(bins[:-1], dtype=np.float32) for _ in range(n_samples)]

        histtype='bar'
        if not stacked:
            histtype='stepfilled'

        plt.hist(binned, bins,
            weights=weights,
            stacked=stacked,
            density=False,
            color=colors,
            label=labels,
            histtype=histtype,
#            linewidth=2,
#            linecolor=colors,
            alpha=yalpha,
            )

        if yalpha:
            plt.hist(binned, bins,
                weights=y,
                stacked=False,
                density=False,
#                label=labels,
                color=colors,
                histtype='step',
                linewidth=2,
                fill=False,
                )

    if signal_colors is None:
        signal_colors = ['orange', 'cyan', 'lime', 'magenta']

    ## plot signals
    n_signals = 0
    if signals is not None:
        n_signals = len(signals)
        signal_colors = signal_colors[:n_signals] if signal_colors else None
        signal_labels = signal_labels[:n_signals] if signal_labels else None
        if stack_signals: # stack signals on top of ytotal
            for i_sig in range(n_signals):
                for j_bin in range(n_bins):
                    signals[i_sig][j_bin] = signals[i_sig][j_bin] + ytotal[j_bin]
        binned = [np.asarray(bins[:-1], dtype=np.float32) for _ in range(n_signals)]
        plt.hist(binned, bins,
            weights=signals,
            stacked=False,
            density=False,
            label=signal_labels,
            color=signal_colors,
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
            color=data_color,
            ecolor=data_color,
            elinewidth=2,
            capsize=0,
            markersize=8,
            zorder=100,
            )

    ## axis limits
    if xlim is not None:
        ax1.set_xlim(*xlim)
    if ylim is not None:
        ax1.set_ylim(*ylim)

    if not xminorticks:
        ax1.minorticks_off()

    ## axis labels
    _label = ''
    if ylabel:
        _label = ylabel
        if unit:
            _label += (' / (%g %s)' % (binwidths[0], unit))
        plt.ylabel(_label, fontsize=fontsize) # HACK
    if not ratio:
        _label = ''
        if xlabel:
            _label = xlabel
            if unit:
                _label += ' [%s]' % (unit)
            plt.xlabel(_label, fontsize=fontsize) # HACK

    if xlabels:
        # We want to show all ticks...
        ax1.set_xticks(np.arange(n_bins))
        # ... and label them with the respective list entries.
        ax1.set_xticklabels(xlabels, fontsize=xlabels_fontsize)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
        # turn spines off
#        for edge, spine in ax1.spines.items():
#            spine.set_visible(False)

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
        else:
            leg_handles = list()
            leg_labels = list()

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
            print('DEBUG: ytotal total_mean = ', total_mean, flush=True)
        elif data is not None:
            sum_data = sum(data)
            total_mean = sum([y_i*x_i/sum_data for y_i, x_i in zip(data, bincenters)])
            print('DEBUG: data total_mean = ', total_mean, flush=True)

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
        if not xminorticks:
            ax2.minorticks_off()

        _label = ''
        if xlabel:
            _label = xlabel
            if unit:
                _label += ' [%s]' % (unit)
            plt.xlabel(_label)

        fig.subplots_adjust(wspace=0, hspace=0)

    fig.align_labels()

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


def make_heatmap(data, 
                 xlabel=None,
                 ylabel=None,
                 xlabels=None,
                 ylabels=None,
                 xlabels_rotation=0,
                 cbar=False,
                 cbar_kw=None, 
                 cbar_label="",
                 cbar_range=None,
                 annotate=False, 
                 figsize=None, 
                 imshow=False,
                 threshold=None,
                 **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Arguments:
        data       : A 2D numpy array of shape (N,M)
    Optional arguments:
        xlabels    : A list or array of length M with the labels
                     for the columns
        ylabels    : A list or array of length N with the labels
                     for the rows
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbar_label  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    
    From: https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    if figsize is None:
        figsize = (12, 12)
    
    if cbar_kw is None:
        cbar_kw = dict()
    
    if xlabels or ylabels:
        imshow = True
    
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    if imshow:
        im = ax.imshow(data, **kwargs)
    else:
        im = ax.pcolormesh(data, **kwargs)

    if cbar_range:
        im.set_clim(*cbar_range)

    if cbar: # Create colorbar
        cbar = fig.colorbar(im, ax=ax, **cbar_kw)
        if cbar_label:
            cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=16)

    if xlabels:
        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(xlabels, fontsize=14)
    
    if ylabels:
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(ylabels, fontsize=14)

    if xlabels:
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment. (-45)
        if xlabels_rotation:
            plt.setp(ax.get_xticklabels(), rotation=xlabels_rotation, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)

    if annotate:
        texts = annotate_heatmap(im, data, valfmt="{x:.0f}", fontsize=14, threshold=threshold)

    ax.tick_params(axis=u'both', which=u'both', length=0)
    
    fig.tight_layout()
    
    return fig, ax



def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.
    Further arguments are passed on to the created text labels.
    """
    
    shift = 0.0 # 0.5 # HACK

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            _n = im.norm(data[i, j])
#            kw.update(color=textcolors[_n > threshold and _n < 0.95]) # HACK
            kw.update(color=textcolors[_n >= threshold])
            text = im.axes.text(j+shift, i+shift, valfmt(int(round(data[i, j])), None), **kw) # HACK
            texts.append(text)

    return texts


def brazil(x, exp, obs,
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
