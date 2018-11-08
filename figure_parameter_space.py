# ---
# jupyter:
#   hide_input: false
#   jupytext_format_version: '1.3'
#   jupytext_formats: ipynb,py:light
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
population_summary_methods = {
    'Mean': mean,
    'Best': amin,
    }

def popmap(M, num_params, weighted, error_func_name,
           pop_summary_name='Best', **kwds):
    global curfig
    # always use the same random seed for cacheing
    seed(34032483)    
    # Set up ranges of variables, and generate arguments to pass to model function
    vx, vy = selected_axes = ('alpha', 'beta')
    pop_summary = population_summary_methods[pop_summary_name]
    error_func = error_functions[error_func_name]
    axis_ranges = dict((k, linspace(*(v+(M,)))) for k, v in kwds.items() if k in selected_axes)
    axis_ranges['temp'] = zeros(num_params)
    array_kwds = meshed_arguments(selected_axes+('temp',), kwds, axis_ranges)
    del array_kwds['temp']
    # Run the model
    res = simple_model(M*M*num_params, array_kwds, use_standalone_openmp=True)
    res = simple_model_results(M*M*num_params, res, error_func, weighted, shape=(M, M, num_params))
    mse = res.mse
    # Analyse the data
    mse = mse*180/pi
    mse_summary = pop_summary(mse, axis=2)
    # Plot the data
    extent = (kwds[vx]+kwds[vy])    
    mse_summary = median_filter(mse_summary, mode='nearest', size=5)
    mse_summary_blur = gaussian_filter(mse_summary, 3, mode='nearest')
    imshow(mse_summary, origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, extent=extent)
    title('Best fits close to overall best fit')
    xlabel(r'Adaptation strength $\alpha$')
    ylabel(r'Onset strength $\beta$')
    cb = colorbar()
    cb.set_label(error_func_name, rotation=270, labelpad=20)
    cs = contour(mse_summary_blur, origin='lower',
                 levels=[15, 30, 45], colors='w',
                 extent=extent)
    clabel(cs, colors='w', inline=True, fmt='%d')
    

def parameter_space(N, M_popmap, num_params,
                    weighted, error_func_name, error_cutoffs,
                    search_params,
                    N_show, transp,
                    interpolate_bmf=True,
                    ):
    # always use the same random seed for cacheing
    seed(34032483)
    # Get simple parameters
    error_func = error_functions[error_func_name]
    # Run the model
    res = simple_model(N, search_params, use_standalone_openmp=True)
    res = simple_model_results(N, res, error_func, weighted=weighted, interpolate_bmf=interpolate_bmf)
    mse = res.mse
    peak_phase = res.peak_phase
    norm_peak_fr = res.norm_measures['peak']
    # Properties of lowest MSE value
    idx_best = argmin(mse)
    best_peak_phase = peak_phase[idx_best, :]
    best_norm_peak_fr = norm_peak_fr[idx_best, :]
    bestvals = []
    for k in search_params.keys():
        v = res.raw.params[k][idx_best]
        bestvals.append('%s=%.2f' % (k, v))
    print 'Best: ' + ', '.join(bestvals)
    
    ############# Plot the data
    curfig = figure(dpi=65, figsize=(8, 3.5))
    
    # We only want to show N_show good peak phase curves, so we apply some criteria
    idx_keep = amax(peak_phase, axis=1)>1*pi/180
    idx_keep = idx_keep & (amin(peak_phase, axis=1)>0)
    idx_keep = idx_keep & (amin(peak_phase, axis=1)<=pi)
    idx_keep = idx_keep & (amax(abs(diff(peak_phase, axis=1)), axis=1)<pi/2)
    idx_keep, = idx_keep.nonzero()
    idx_keep = idx_keep[:N_show]
    # Plot the extracted phase curves
    subplot(121)
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
    xticks(dietz_fm/Hz)
    xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')
    # Plot the best fits
    subplot(122)
    popmap(M=M_popmap, num_params=num_params,
           weighted=weighted, error_func_name=error_func_name,
           **search_params)
    title('Best fits')
    
    tight_layout()

    # Label panels
    for i, c in enumerate('AB'):
        text(0.48*i, .92, c,
             transform=gcf().transFigure, fontsize=16)

        
# N = 1000; M_popmap=10; num_params=20 # quick, low quality
#N = 10000; M_popmap=20; num_params=100 # medium quality
N = 10000; M_popmap=80; num_params=500 # high quality (will take many hours to run)

search_params = dict(
    taui_ms=(0.1, 10), taue_ms=(0.1, 10), taua_ms=(0.1, 10),
    level=(-25, 25), alpha=(0, 0.99), beta=(0, 2),
    gamma=(0.1, 1))

parameter_space(N=N, M_popmap=M_popmap, num_params=num_params,
                weighted=False, error_func_name="Max error",
                error_cutoffs=[15, 30, 45],
                N_show=1000, transp=0.1,
                search_params=search_params,
               )
savefig('figure_parameter_space.pdf')

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
    
def look_for_correlations(N, search_params,
                          weighted=False, error_func_name="Max error",
                          max_error=30, plotmode='scatter',
                         ):
    if plotmode=='density':
        density_plotter = plot_density_map
    elif plotmode=='independent_density':
        plotmode = 'density'
        density_plotter = plot_independent_density_map
    # always use the same random seed for cacheing
    seed(34032483)
    # Get simple parameters
    error_func = error_functions[error_func_name]
    # Run the model
    res = simple_model(N, search_params, use_standalone_openmp=True)
    res = simple_model_results(N, res, error_func, weighted=weighted, interpolate_bmf=False)
    mse = res.mse
    meanvs = mean(res.raw_measures['vs'], axis=1)
    good_indices = mse<max_error*pi/180
    regions = [('All', good_indices, 'blue')]
#     regions = [('Low VS', good_indices & (meanvs < 0.75), 'red'),
#                ('High VS', good_indices & (meanvs >= 0.75), 'blue'),
#               ]
    # plot a histogram to check we're good
    figure(figsize=(8, 8), dpi=85)
    nparam = len(res.raw.params)
    gs = GridSpec(nparam-1, nparam-1, wspace=0, hspace=0)
    #for (px, vx), (py, vy) in itertools.combinations(res.raw.params.items(), 2):
    for i in xrange(nparam):
        for j in xrange(i+1, nparam):
            px = res.raw.params.keys()[i]
            py = res.raw.params.keys()[j]
            vx = res.raw.params[px]
            vy = res.raw.params[py]
            subplot(gs[j-1, i])
            for condname, cond, condcol in regions:
                rvx = vx[cond]
                rvy = vy[cond]
                if plotmode=='scatter':
                    plot(rvx, rvy, ',', c=condcol)
                elif plotmode=='density':
                    density_plotter(rvx, rvy, 40,
                                    xmin=search_params[px][0],
                                    xmax=search_params[px][1],
                                    ymin=search_params[py][0],
                                    ymax=search_params[py][1],
                                    )
            xlim(*search_params[px])
            ylim(*search_params[py])
            if j==nparam-1:
                xlabel(latex_parameter_names[px])
                xticks(search_params[px])
                ticklabels = gca().get_xticklabels()
                ticklabels[0].set_ha('left')
                ticklabels[-1].set_ha('right')
            else:
                xticks([])
            if i==0:
                ylabel(latex_parameter_names[py])
                yticks(search_params[py])
                ticklabels = gca().get_yticklabels()
                ticklabels[0].set_va('bottom')
                ticklabels[-1].set_va('top')
            else:
                yticks([])
    tight_layout()

for plotmode in ['scatter', 'density', 'independent_density']:
    look_for_correlations(N=50000, search_params=search_params, plotmode=plotmode)
