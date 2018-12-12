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

# +
# %matplotlib notebook
from brian2 import *
from collections import OrderedDict
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib.gridspec import GridSpecFromSubplotSpec
import joblib
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter, median_filter, minimum_filter
from simple_model import *
from model_explorer_jupyter import meshed_arguments
import itertools
import numba

def normed(X, *args):
    m = max(amax(abs(Y)) for Y in (X,)+args)
    return X/m

def rmse(x, y, axis=1):
    return sqrt(mean((x-y)**2, axis=axis))

def maxnorm(x, y, axis=1):
    return amax(abs(x-y), axis=axis)

error_functions = {
    'RMS error': rmse,
    'Max error': maxnorm,
    }

latex_parameter_names = dict(
    taue_ms=r"$\tau_e$ (ms)",
    taui_ms=r"$\tau_i$ (ms)",
    taua_ms=r"$\tau_a$ (ms)",
    alpha=r"$\alpha$",
    beta=r"$\beta$",
    gamma=r"$\gamma$",
    level=r"$L$ (dB)",
    )

# +
def single_neuron_predictions(N, search_params,
                              N_show=1000, transp=0.1,
                              weighted=False, error_func_name="Max error",
                              max_error=30,
                              ):
    search_params_all = search_params
    search_params = dict((k, v) for k, v in search_params.items() if isinstance(v, tuple))
    # always use the same random seed for cacheing
    seed(34032483)
    # Get simple parameters
    error_func = error_functions[error_func_name]
    # Run the model
    res = simple_model(N, search_params_all, use_standalone_openmp=True, update_progress='text')
    res = simple_model_results(N, res, error_func, weighted=weighted, interpolate_bmf=False)
    good_indices = res.mse<max_error*pi/180
    Ngood = sum(good_indices)
    print "Found %d good results" % Ngood
    # Make predictions for different gains
    curparams = dict()
    for k, v in res.raw.params.items():
        curparams[k] = v[good_indices]
    baselevel = curparams['level']
    results = []
    for i, gain in enumerate([-40, -20, 20, 40]):
        curparams['level'] = baselevel+gain
        res = simple_model(Ngood, curparams, use_standalone_openmp=True, update_progress='text')
        res = simple_model_results(Ngood, res, error_func, weighted=weighted, interpolate_bmf=False)
        peak_phase = res.peak_phase
        idx_keep = amin(peak_phase, axis=1)>0
        idx_keep = idx_keep & (baselevel+gain<60)
        idx_keep = idx_keep & (sum(res.raw.mean_fr, axis=1)<1e10) # discard numerically unstable results
        print 'For gain %d, kept %d results (%.1f%%)'%(gain, sum(idx_keep), sum(idx_keep)*100.0/Ngood)
        peak_phase = peak_phase[idx_keep, :]
        unrolled_peak_phase = peak_phase[:, 0][:, newaxis]+cumsum(hstack((zeros((peak_phase.shape[0], 1)), log(exp(1j*diff(peak_phase, axis=1))).imag)), axis=1)
        results.append((i, gain, res.peak_phase, unrolled_peak_phase))
    figure(figsize=(9, 4))
    subplot(121)
    for i, gain, peak_phase, unrolled_peak_phase in results:
        # compute circular stats
        m = sum(exp(1j*peak_phase), axis=0)/peak_phase.shape[0]
        mean_phase = log(m).imag
        std_phase = sqrt(-2*log(abs(m)))
        plot(dietz_fm/Hz, mean_phase*180/pi, c='C'+str(i), label='%+d dB'%gain)
        #errorbar(dietz_fm/Hz+gain/30., mean_phase*180/pi, std_phase*180/pi, c='C'+str(i), label='%+d dB'%gain)
    errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--r', label='Data')
    legend(loc='best')
    grid()
    ylim(0, 360)
    yticks([0, 90, 180, 270, 360])
    xticks(dietz_fm/Hz)
    xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')
    for i, gain, peak_phase, unrolled_peak_phase in results:
        unrolled_peak_phase = unrolled_peak_phase[:N_show, :]
        subplot(2, 4, 3+(i%2)+(i//2)*4)
        plot(dietz_fm/Hz, unrolled_peak_phase.T*180/pi, '-', color='C'+str(i), alpha=transp)
        plot(dietz_fm/Hz, unrolled_peak_phase.T*180/pi+360, '-', color='C'+str(i), alpha=transp)
        plot(dietz_fm/Hz, unrolled_peak_phase.T*180/pi-360, '-', color='C'+str(i), alpha=transp)
        grid()
        ylim(0, 360)
        yticks([0, 90, 180, 270, 360])
        xticks(dietz_fm/Hz)
        title('%+d dB'%gain)
        errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--r', label='Data')
        
    tight_layout()
    
        
search_params = dict(
    taui_ms=(0.1, 10), taue_ms=(0.1, 10), taua_ms=(0.1, 10),
    level=(-25, 25), alpha=(0, 0.99), beta=(0, 2),
    gamma=(0.1, 1))


single_neuron_predictions(N=100000, search_params=search_params, N_show=200)

savefig('figure_single_neuron_predictions.pdf')
