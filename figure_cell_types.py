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

# # Cell types

# +
# %matplotlib notebook
from brian2 import *
from collections import OrderedDict
from matplotlib import cm
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from simple_model import *
from model_explorer_jupyter import meshed_arguments

def normed(X, *args):
    m = max(amax(abs(Y)) for Y in (X,)+args)
    return X/m

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

latex_parameter_names = dict(
    taue_ms=r"$\tau_e$ (ms)",
    taui_ms=r"$\tau_i$ (ms)",
    taua_ms=r"$\tau_a$ (ms)",
    alpha=r"$\alpha$",
    beta=r"$\beta$",
    gamma=r"$\gamma$",
    level=r"$L$ (dB)",
    )

# ...

# To do:
# - Define parameter range
# - Compute popmap over this parameter range
# - From that we can compute rMTF and tMTF and error
# - Divide param space into regions based on the above
# - Generate parameter histograms
# - Select representative examples and plot them in more detail

if 0:
    population_summary_methods = {
        'Mean': mean,
        'Best': amin,
        }

    def popmap(M, num_params, blur_width, 
               weighted, error_func_name,
               pop_summary_name='Best', smoothing=True,
               **kwds):
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
        res = simple_model(M*M*num_params, array_kwds, update_progress='text')
        res = simple_model_results(M*M*num_params, res, error_func, weighted, shape=(M, M, num_params))
        mse = res.mse
        # Analyse the data
        mse = mse*180/pi
        mse_summary = pop_summary(mse, axis=2)
        # Plot the data
        if smoothing:
            mse_summary = gaussian_filter(mse_summary, blur_width*M, mode='nearest')
            mse_summary = zoom(mse_summary, 100./M, order=1)
            mse_summary = gaussian_filter(mse_summary, blur_width*100., mode='nearest')
        extent = (kwds[vx]+kwds[vy])

        imshow(mse_summary, origin='lower left', aspect='auto',
               interpolation='nearest', vmin=0, extent=extent)
        title('Best fits close to overall best fit')
        xlabel(r'Adaptation strength $\alpha$')
        ylabel(r'Inhibition strength $\beta$')
        cb = colorbar()
        cb.set_label(error_func_name, rotation=270, labelpad=20)
        cs = contour(mse_summary, origin='lower',
                     levels=[15, 30, 45], colors='w',
                     extent=extent)
        clabel(cs, colors='w', inline=True, fmt='%d')

# +
def amin_from_to(arr_from, arr_to):
    i, j = mgrid[:arr_from.shape[0], :arr_from.shape[1]]
    k = argmin(arr_from, axis=2)
    return arr_to[i, j, k]

population_summary_methods = {
    'Mean': lambda arr_from, arr_to: mean(arr_to, axis=2),
    'Best': amin_from_to,
    }

def plot_cell_types(M, num_params, params,
                    weighted, error_func_name,
                    pop_summary_name='Best'):
    # always use the same random seed for cacheing
    seed(34032483)    
    # Set up ranges of variables, and generate arguments to pass to model function
    vx, vy = selected_axes = ('alpha', 'beta')
    pop_summary = population_summary_methods[pop_summary_name]
    error_func = error_functions[error_func_name]
    axis_ranges = dict((k, linspace(*(v+(M,)))) for k, v in params.items() if k in selected_axes)
    axis_ranges['temp'] = zeros(num_params)
    array_kwds = meshed_arguments(selected_axes+('temp',), params, axis_ranges)
    del array_kwds['temp']
    # Run the model
    res = simple_model(M*M*num_params, array_kwds, update_progress='text')
    res = simple_model_results(M*M*num_params, res, error_func, weighted,
                               interpolate_bmf=True, shape=(M, M, num_params))
    mse = res.mse
    # Analyse the data
    mse = mse*180/pi
    mse_summary = pop_summary(mse, mse)
    # Plot the data
    fig = figure(figsize=(5, 8))
    gs_maps = GridSpec(2, 3, bottom=.75, top=1)
    gs_params = GridSpec(2, 5, bottom=0.5, top=0.75)
    gs_hist = GridSpec(2, 4, bottom=0.25, top=0.5)
    gs_ex = GridSpec(1, 4, bottom=0.0, top=0.25)
    ordered_gridspecs = [gs_maps, gs_params, gs_hist, gs_ex]
    
    # temporary global layout
    for gs in ordered_gridspecs:
        h, w = gs.get_geometry()
        for i in range(w):
            for j in range(h):
                subplot(gs[j, i])
    
    # Maps panel
    extent = (params[vx]+params[vy])
    subplot(gs_maps[0, 0]) # error
    imshow(mse_summary, origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, extent=extent)
    subplot(gs_maps[1, 0]) # vector strength
    imshow(pop_summary(mse, mean(res.raw_measures['vs'], axis=3)),
           origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, vmax=1, extent=extent)
    subplot(gs_maps[0, 1]) # rBMF
    imshow(pop_summary(mse, res.bmf['mean']),
           origin='lower left', aspect='auto',
           interpolation='nearest', vmin=4, vmax=64, extent=extent)
    subplot(gs_maps[1, 1]) # tBMF
    imshow(pop_summary(mse, res.bmf['vs']),
           origin='lower left', aspect='auto',
           interpolation='nearest', vmin=4, vmax=64, extent=extent)
    subplot(gs_maps[0, 2]) # rMD
    imshow(pop_summary(mse, res.moddepth['mean']),
           origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, vmax=1, extent=extent)
    subplot(gs_maps[1, 2]) # tMD
    imshow(pop_summary(mse, res.moddepth['vs']),
           origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, vmax=1, extent=extent)
    
    # tight layout
    for i, gs in enumerate(ordered_gridspecs[::-1]):
       gs.tight_layout(fig,
            rect=[0, 1.0*i/len(ordered_gridspecs), 1, 1.0*(i+1)/len(ordered_gridspecs)])
    
    
plot_cell_types(
    M=10, num_params=20,
    weighted=False, error_func_name='Max error',
    params=dict(
        taui_ms=(0.1, 10), taue_ms=(0.1, 10), taua_ms=(0.1, 10),
        level=(-25, 25), alpha=(0, 0.99), beta=(0, 2),
        gamma=(0.1, 1)),
    )

show()
