# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# # Cell types

# +
# %matplotlib notebook
from brian2 import *
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import matplotlib.patches as patches
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter, median_filter
import numba
import numpy

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

# +
def where_close_to_best(mse, max_error):
    #mse_best = amin(mse, axis=2)
    #return abs(mse-mse_best[:, :, newaxis])<max_error
    return mse<max_error

def fraction_close_to_best(whereclose):
    return sum(whereclose, axis=2)*1.0/whereclose.shape[2]

def desaturate(img, saturation):
    # assume img is rgb shape (M, M, 3)
    img = rgb_to_hsv(img)
    img[:, :, 1] *= saturation
    img = hsv_to_rgb(img)
    return img

def pop_summary(mse, arr, max_error=30, vmin=None, vmax=None):
    # if arr is mse just return the best
    if mse is arr:
        return amin(mse, axis=2)
    # compute where is close to best
    whereclose = where_close_to_best(mse, max_error)
    # compute mean across close values
    arr_mean = (1.0*sum(arr*whereclose, axis=2))/sum(whereclose, axis=2)
    # normalise range
    if vmin is None:
        vmin = nanmin(arr_mean)
    if vmax is None:
        vmax = nanmax(arr_mean)
    arr_mean = (arr_mean-vmin)/(vmax-vmin)
    # compute range across close values and all values
    arr_nans = arr.copy()
    arr_nans[~whereclose] = nan
    arr_std_close = nanmax(arr_nans, axis=2) - nanmin(arr_nans, axis=2)
    arr_std_all = amax(arr, axis=2)-amin(arr, axis=2)
    saturation = 1-arr_std_close/arr_std_all
    # convert to rgb
    img = cm.viridis(arr_mean)[:, :, :3] # discard alpha
    # desaturate image
    img = desaturate(img, saturation**2)
    # hide values where nothing is good
    img = dstack((img, sum(whereclose, axis=2)>0))
    return img

def plot_cell_types(M, num_params, params,
                    weighted, error_func_name,
                    num_examples_to_show=20, example_alpha=.2,
                    ):
    # always use the same random seed for cacheing
    seed(34032483)
    # Set up ranges of variables, and generate arguments to pass to model function
    vx, vy = selected_axes = ('alpha', 'beta')
    error_func = error_functions[error_func_name]
    axis_ranges = dict((k, linspace(*(v+(M,)))) for k, v in params.items() if k in selected_axes)
    axis_ranges['temp'] = zeros(num_params)
    array_kwds = meshed_arguments(selected_axes+('temp',), params, axis_ranges)
    del array_kwds['temp']
    # Run the model
    res = simple_model(M*M*num_params, array_kwds, update_progress='text', use_standalone_openmp=True)
    res = simple_model_results(M*M*num_params, res, error_func, weighted,
                               interpolate_bmf=True, shape=(M, M, num_params))
    # Analyse the data
    mse = res.mse
    mse = mse*180/pi
    mse_summary = pop_summary(mse, mse, vmin=0)
    # Define regions
    meanvs = mean(res.raw_measures['vs'], axis=3)
    tMD = res.moddepth['mean']
    rBMF = res.bmf['mean']
    regions = [('All', mse < 30, 'lightgray', 1),
               ('Low VS', logical_and(mse < 30, meanvs < 0.75), 'red', 0.25),
               ('High VS', (mse < 30) & (meanvs >= 0.75), 'blue', 0.25),
              ]
    
    # Plot the data
    fig = figure(figsize=(10, 6))
    gs_hist = GridSpec(3, 4, left=.05, bottom=0.35, top=1)
    gs_ex = GridSpec(1, 4, left=.05, bottom=0.0, top=0.34)
    ordered_gridspecs = [gs_hist, gs_ex]

    def hatchback():
        p = patches.Rectangle((extent[0], extent[2]), extent[1]-extent[0], extent[3]-extent[2],
                              hatch='xxxx', fill=True, fc=(0.9,)*3, ec=(0.8,)*3, zorder=-10)
        gca().add_patch(p)

    # Property maps
    cell_properties = dict([
        ('tMTF', (meanvs, 0, 1)),
        ('tMD', (res.moddepth['vs'], 0, 1)),
        ('tBMF', (res.bmf['vs'], 4, 64)),
        ('rMD', (res.moddepth['mean'], 0, 1)),
        ('rBMF', (res.bmf['mean'], 4, 64)),
        ])

    # Region examples
    ax_lf = subplot(gs_ex[0, 0])
    ax_hf = subplot(gs_ex[0, 1])
    ax_rmtf = subplot(gs_ex[0, 2])
    ax_tmtf = subplot(gs_ex[0, 3])
    ax_lf.set_title(r'$f_m=4$ Hz')
    ax_hf.set_title(r'$f_m=64$ Hz')
    ax_rmtf.set_title('rMTF')
    ax_tmtf.set_title('tMTF')
    phase = linspace(0, 2*pi, 100)
    env = 0.5*(1-cos(phase))
    region_example_params = {}
    for ax in [ax_lf, ax_hf]:
        ax.fill_between(phase*180/pi, 0, env, color=(0.9,) * 3, zorder=-2)
        ax.set_xlabel('Phase (deg')
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlim(0, 360)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
    for ax in [ax_rmtf, ax_tmtf]:
        ax.set_xlabel(r'$f_m$')
        ax.set_xticks([4, 8, 16, 32, 64])
        ax.set_ylim(0, 1)
    seed(54383278)
    for region_name, cond, col, alpha in regions[1:]:
        print "Region %s contains %.1f%% of good parameters" % (region_name, sum(cond)*100.0/sum(mse<30))
        # Construct parameter values for that region
        example_indices = arange(sum(cond))
        shuffle(example_indices)
        for cur_ex, ex_idx in enumerate(example_indices[:num_examples_to_show+1]):
            if cur_ex<num_examples_to_show:
                ex_alpha = example_alpha
                lw = 1
            else:
                ex_alpha = 1
                lw = 2
                all_values = []
                for paramname in params.keys():
                    low, high = params[paramname]
                    values = reshape(res.raw.params[paramname], (M, M, num_params))
                    values = (values[cond]-high)/(high-low)
                    all_values.append(values)
                all_values = array(all_values) # shape (num_params, sum(cond))
                # too large to compute in memory for largest simulation, so use numba
                #ex_idx = argmin(sum(sum((all_values[:, :, newaxis]-all_values[:, newaxis, :])**2, axis=0), axis=1))
                output = zeros(all_values.shape[1])
                @numba.jit(nopython=True)
                def ex_idx_reduction(all_values, output):
                    for i in range(all_values.shape[0]):
                        for j in range(all_values.shape[1]):
                            for k in range(all_values.shape[1]):
                                output[k] += (all_values[i, j]-all_values[i, k])**2
                ex_idx_reduction(all_values, output)
                ex_idx = argmin(output)
            region_params = {}        
            for paramname in params.keys():
                values = reshape(res.raw.params[paramname], (M, M, num_params))
                values = values[cond]
                region_params[paramname] = values[ex_idx]
            cur_res = simple_model(1, region_params, record=['out'])
            cur_res = simple_model_results(1, cur_res, error_func, weighted, interpolate_bmf=True)
            if cur_ex==num_examples_to_show:
                print '    most representative example error=%.1f deg, params = %s' % (cur_res.mse[0]*180/pi, region_params)
                region_params['tMTF'] = mean(cur_res.raw_measures['vs'])
                region_params['tMD'] = cur_res.moddepth['vs'][0]
                region_params['tBMF'] = cur_res.bmf['vs'][0]
                region_params['rMD'] = cur_res.moddepth['mean'][0]
                region_params['rBMF'] = cur_res.bmf['mean'][0]
                region_example_params[region_name] = region_params
            out = cur_res.raw.out
            t = cur_res.raw.t
            for j, ax in [(0, ax_lf), (-1, ax_hf)]:
                I = logical_and(t>=cur_res.raw.start_time[j], t<cur_res.raw.end_time[j])
                phase = ((2*pi*dietz_fm[j]*t[I])%(2*pi))*180/pi
                ax.plot(phase, normed(out[0, j, I]), c=col, alpha=ex_alpha, lw=lw)
            ax_rmtf.plot(dietz_fm, cur_res.norm_measures['mean'].T, c=col, alpha=ex_alpha, lw=lw)
            ax_tmtf.plot(dietz_fm, cur_res.raw_measures['vs'].T, c=col, alpha=ex_alpha, lw=lw)

        
    # Histograms
    for i, paramname in enumerate(['alpha', 'beta', 'gamma', 'level', 'taue_ms', 'taui_ms', 'taua_ms',
                                   'tMTF', 'tMD', 'tBMF', 'rMD', 'rBMF']):
        subplot(gs_hist[i//4, i%4])
        if paramname in res.raw.params:
            values = reshape(res.raw.params[paramname], (M, M, num_params))
            low, high = params[paramname]
        else:
            values, low, high = cell_properties[paramname]
        for condname, cond, col, alpha in regions:
            v = values[cond]
            hist(v, bins=M, range=(low, high), histtype='stepfilled',
                 fc=col, alpha=alpha, label=condname)
            if condname in region_example_params and paramname in region_example_params[condname]:
                axvline(region_example_params[condname][paramname], c=col)
        xlim(low, high)
        yticks([])
        title(latex_parameter_names.get(paramname, paramname))

    # Tight layout
    for gs in ordered_gridspecs:
        gs.tight_layout(fig, rect=(gs.left, gs.bottom, gs.right, gs.top))

    # annotate
    for c, loc in zip('AB', [.98, .33]):
        text(0.02, loc, c, fontsize=14, transform=fig.transFigure,
             horizontalalignment='left', verticalalignment='top')

    
plot_cell_types(
    #M=20, num_params=100, # 40k parameter sets total, 2m
    #M=40, num_params=100, # 160k parameter sets total, 5-10m
    M=80, num_params=500, # 3.2M parameter sets total, several hours
    num_examples_to_show=10,
    example_alpha=.2,
    weighted=False, error_func_name='Max error',
    params=dict(
        taui_ms=(0.1, 10), taue_ms=(0.1, 10), taua_ms=(0.1, 10),
        level=(-25, 25), alpha=(0, 0.99), beta=(0, 2),
        gamma=(0.1, 1)),
    )

savefig('figure_cell_types.pdf')
show()
