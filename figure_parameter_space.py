# ---
# jupyter:
#   hide_input: false
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.1
#   jupytext_format_version: '1.3'
#   jupytext_formats: py:light
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 2
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython2
#     version: 2.7.15
# ---

# # Parameter space figure

# +
# %matplotlib notebook
from brian2 import *
from collections import OrderedDict
from scipy.interpolate import interp1d
from matplotlib import cm
import joblib
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter, median_filter
from simple_model import *
from model_explorer_jupyter import meshed_arguments
import itertools
import numba

def normed(X, *args):
    m = max(amax(abs(Y)) for Y in (X,)+args)
    return X/m
# -

# Error functions

# +
def rmse(x, y, axis=1):
    return sqrt(mean((x-y)**2, axis=axis))

def maxnorm(x, y, axis=1):
    return amax(abs(x-y), axis=axis)

error_functions = {
    'RMS error': rmse,
    'Max error': maxnorm,
    }
# -

# Parameter names

latex_parameter_names = dict(
    taue_ms=r"$\tau_e$ (ms)",
    taui_ms=r"$\tau_i$ (ms)",
    taua_ms=r"$\tau_a$ (ms)",
    alpha=r"$\alpha$",
    beta=r"$\beta$",
    gamma=r"$\gamma$",
    level=r"$L$ (dB)",
    )

# Run analysis and plot figure

# +
from scipy import stats

# # Take a series of x, y points and plot a density map using kernel density estimation
# # N is the grid size for the density image

def density_map(x, y, N, xmin=None, xmax=None, ymin=None, ymax=None):
    # Peform the kernel density estimate
    if xmin is None:
        xmin = amin(x)
    if xmax is None:
        xmax = amax(x)
    if ymin is None:
        ymin = amin(y)
    if ymax is None:
        ymax = amax(y)
    xx, yy = mgrid[xmin:xmax:N*1j, ymin:ymax:N*1j]
    positions = vstack([xx.ravel(), yy.ravel()])
    values = vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    extent = (xmin, xmax, ymin, ymax)
    return f.T, extent

def plot_density_map(x, y, N, xmin=None, xmax=None, ymin=None, ymax=None, **args):
    img, extent = density_map(x, y, N, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    imshow(img, origin='lower left', aspect='auto', interpolation='nearest',
           extent=extent,
           vmin=0, vmax=amax(img),#/0.7,
           **args
           )

# Take a series of x, y points and plot a density map using kernel density estimation
# N is the grid size for the density image. For the independent density map it shows
# the ratio of the density of the joint density to what you'd expect if the variables
# were independent

def independent_density_map(x, y, N, xmin=None, xmax=None, ymin=None, ymax=None):
    # Peform the kernel density estimate
    if xmin is None:
        xmin = amin(x)
    if xmax is None:
        xmax = amax(x)
    if ymin is None:
        ymin = amin(y)
    if ymax is None:
        ymax = amax(y)
    xx, yy = mgrid[xmin:xmax:N*1j, ymin:ymax:N*1j]
    xxx = linspace(xmin, xmax, N)
    yyy = linspace(ymin, ymax, N)
    positions = vstack([xx.ravel(), yy.ravel()])
    values = vstack([x, y])
    kernel = stats.gaussian_kde(values)
    kernelx = stats.gaussian_kde(x)
    kernely = stats.gaussian_kde(y)
    f = log(np.reshape(kernel(positions).T, xx.shape)/(kernelx(xxx)[newaxis, :]*kernely(yyy)[:, newaxis]))
    extent = (xmin, xmax, ymin, ymax)
    return f.T, extent

def plot_independent_density_map(x, y, N, xmin=None, xmax=None, ymin=None, ymax=None, **args):
    img, extent = independent_density_map(x, y, N, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    imshow(img, origin='lower left', aspect='auto', interpolation='nearest',
           extent=extent, cmap=cm.coolwarm,
           vmin=-2, vmax=2,
           **args
           )
    
def parameter_space(N, search_params, N_show=1000, transp=0.1,
                    weighted=False, error_func_name="Max error",
                    max_error=30, plotmode='scatter',
                    error_upper_cutoff=90,
                    ):
    figtitle = plotmode
    if plotmode=='density':
        density_plotter = plot_density_map
    elif plotmode=='independent_density':
        plotmode = 'density'
        density_plotter = plot_independent_density_map
    elif plotmode=='error':
        # aim for at least 100 params per pixel, but only need 40x40 max pixels
        N_img = min(int(sqrt(N/100.)), 40)
        print 'Image dimension {N_img}x{N_img}, approximately {samp} samples per pixel'.format(
                    N_img=N_img, samp=int(N/N_img**2))
    # always use the same random seed for cacheing
    seed(34032483)
    # Get simple parameters
    error_func = error_functions[error_func_name]
    # Run the model
    res = simple_model(N, search_params, use_standalone_openmp=True, update_progress='text')
    res = simple_model_results(N, res, error_func, weighted=weighted, interpolate_bmf=False)
    mse = res.mse
    meanvs = mean(res.raw_measures['vs'], axis=1)
    good_indices = mse<max_error*pi/180
    regions = [('All', good_indices, 'blue')]
    # Plot parameter pairs
    figure(figsize=(8, 8.5), dpi=85)
    #suptitle(figtitle)
    nparam = len(res.raw.params)
    gs = GridSpec(nparam-1, nparam-1, wspace=0, hspace=0)
    image_axes = []
    for i in xrange(nparam):
        for j in xrange(i+1, nparam):
            px = res.raw.params.keys()[i]
            py = res.raw.params.keys()[j]
            vx = res.raw.params[px]
            vy = res.raw.params[py]
            xmin, xmax = search_params[px]
            ymin, ymax = search_params[py]
            image_axes.append(subplot(gs[j-1, i]))
            if plotmode=='error':
                error = 2*pi*ones((N_img, N_img))
                @numba.jit(nopython=True)
                def find_best_error(error, vx, vy, mse, xmin, xmax, ymin, ymax):
                    N = error.shape[0]
                    wx = float(xmax-xmin)/N
                    wy = float(ymax-ymin)/N
                    for k in range(mse.shape[0]):
                        x = vx[k]
                        y = vy[k]
                        i = int((x-xmin)/wx)
                        j = int((y-ymin)/wy)
                        if mse[k]<error[i, j]:
                            error[i, j] = mse[k]
                find_best_error(error, vx, vy, mse, xmin, xmax, ymin, ymax)
                error = error.T*180/pi
                error = median_filter(error, size=5, mode='nearest')
                error_blur = gaussian_filter(error, 3, mode='nearest')
                img_obj = imshow(error, extent=(xmin, xmax, ymin, ymax),
                       origin='lower left', aspect='auto', interpolation='nearest',
                       vmin=0, vmax=error_upper_cutoff, cmap=cm.viridis)
                cs = contour(error_blur, origin='lower',
                             levels=[15, 30, 45], colors='w',
                             extent=(xmin, xmax, ymin, ymax))
                # this weird hack works around a bug in contour
                for level, linecol in zip(cs.levels, cs.collections):
                    if level==15:
                        linecol.set_linestyle('solid')
                    if level==30:
                        linecol.set_linestyle('dashed')
                    if level==45:
                        linecol.set_linestyle('dotted')
            else:
                for condname, cond, condcol in regions:
                    rvx = vx[cond]
                    rvy = vy[cond]
                    if plotmode=='scatter':
                        plot(rvx, rvy, ',', c=condcol)
                    elif plotmode=='density':
                        density_plotter(rvx, rvy, 40,
                                        xmin=xmin, ymin=ymin,
                                        xmax=xmax, ymax=ymax,
                                        )
            xlim(*search_params[px])
            ylim(*search_params[py])
            if j==nparam-1:
                xlabel(latex_parameter_names[px])
                xticks(search_params[px], ['%.2f' % paramval for paramval in search_params[px]])
                ticklabels = gca().get_xticklabels()
                ticklabels[0].set_ha('left')
                ticklabels[0].set_text(' '+ticklabels[0].get_text())
                ticklabels[-1].set_ha('right')
                ticklabels[-1].set_text(ticklabels[-1].get_text()+' ')
                #print ticklabels[0]
                #print ticklabels
                gca().set_xticklabels(ticklabels)
            else:
                xticks([])
            if i==0:
                yticks(search_params[py], ['%.2f' % paramval for paramval in search_params[py]])
                ticklabels = gca().get_yticklabels()
                ticklabels[0].set_va('bottom')
                ticklabels[-1].set_va('top')
                ylabel(latex_parameter_names[py])
                gca().get_yaxis().set_label_coords(-0.2,0.5)
            else:
                yticks([])
    
    # Plot some sample extracted phase curves
    peak_phase = res.peak_phase
    # Properties of lowest MSE value
    idx_best = argmin(mse)
    best_peak_phase = peak_phase[idx_best, :]
    bestvals = []
    for k in search_params.keys():
        v = res.raw.params[k][idx_best]
        bestvals.append('%s=%.2f' % (k, v))
    print 'Best: ' + ', '.join(bestvals)
    # We only want to show N_show good peak phase curves, so we apply some criteria
    idx_keep = amax(peak_phase, axis=1)>1*pi/180
    idx_keep = idx_keep & (amin(peak_phase, axis=1)>0)
    idx_keep = idx_keep & (amin(peak_phase, axis=1)<=pi)
    idx_keep = idx_keep & (amax(abs(diff(peak_phase, axis=1)), axis=1)<pi/2)
    idx_keep, = idx_keep.nonzero()
    idx_keep = idx_keep[:N_show]
    # Plot the extracted phase curves
    subplot(gs[0:2, nparam-3:nparam-1])
    plot(dietz_fm/Hz, peak_phase[idx_keep, :].T*180/pi, '-', color=(0.4, 0.7, 0.4, transp), label='Model (all)')
    plot(dietz_fm/Hz, best_peak_phase*180/pi, '-ko', lw=2, label='Model (best)')
    errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--r', label='Data')
    handles, labels = gca().get_legend_handles_labels()
    lab2hand = OrderedDict()
    for h, l in zip(handles, labels):
        lab2hand[l] = h
    legend(lab2hand.values(), lab2hand.keys(), loc='upper left')
    grid()
    ylim(0, 180)
    yticks([0, 45, 90, 135, 180])
    xticks(dietz_fm/Hz)
    xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')
        
    subplots_adjust(left=0.12, right=0.98, bottom=0.05, top=0.98)
    
    cb = colorbar(img_obj, ax=gcf().axes, use_gridspec=True,
             ticks=range(0, 121, 15),
             orientation='horizontal',
             fraction=0.04, pad=0.08, aspect=40,
            )
    cb.set_label(error_func_name)#, rotation=270, labelpad=20)
    
    for c, xpos in [('A', 0.02), ('B', 0.55)]:
        text(xpos, 0.98, c, fontsize=18, transform=gcf().transFigure,
             horizontalalignment='left', verticalalignment='top')

        
search_params = dict(
    taui_ms=(0.1, 10), taue_ms=(0.1, 10), taua_ms=(0.1, 10),
    level=(-25, 25), alpha=(0, 0.99), beta=(0, 2),
    gamma=(0.1, 1))

#for plotmode in ['error', 'scatter', 'density', 'independent_density']:
#parameter_space(N=50000, search_params=search_params, plotmode='error')
parameter_space(N=800000, search_params=search_params, plotmode='error')

savefig('figure_parameter_space.pdf')
