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

# # Cell types figure (unfinished)

# +
# %matplotlib notebook
from simple_model import *
from brian2 import *
from model_explorer_jupyter import *
import ipywidgets as ipw
from collections import OrderedDict
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib.gridspec import GridSpecFromSubplotSpec
import joblib
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter

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

# ...

# +
N = 10000
error_func = maxnorm
params = OrderedDict([
    ('taui_ms', (0.1, 10)),
    ('taue_ms', (0.1, 10)),
    ('taua_ms', (0.1, 10)),
    ('level', (-25, 25)),
    ('alpha', (0, 0.99)),
    ('beta', (0, 2)),
    ('gamma', (0.1, 1)),
    ])
epsilon = 0.6

raw_res = simple_model(N, params, update_progress='text')
res = simple_model_results(N, raw_res, error_func, weighted=False, interpolate_bmf=True)
keep_index = res.mse<30*pi/180
all_values = raw_res.params.copy()
all_values.update(dict(('bmf/'+k, v) for k, v in res.bmf.items()))
all_values.update(dict(('moddepth/'+k, v) for k, v in res.moddepth.items()))
all_values['mean_vs'] = mean(res.raw_measures['vs'], axis=1)
all_values['mean_onsettiness'] = mean(res.raw_measures['onsettiness'], axis=1)
if 0:
    num_rows = int(ceil(len(all_values)/5.0))
    figure(figsize=(10, 2*num_rows))
    for i, (k, v) in enumerate(all_values.items()):
        subplot(num_rows, 5, i+1)
        hist(v[keep_index])
        title(k)
    tight_layout()
# create keep values
keep_values = OrderedDict()
for k, v in all_values.items():
    keep_values[k] = v[keep_index]
# create histogram-normalised values
norm_values = OrderedDict()
for k, v in keep_values.items():
    i = argsort(v)
    j = zeros_like(i)
    j[i] = arange(len(v))
    norm_values[k] = j*1.0/(len(v)-1)
# find representative examples
Y = vstack(keep_values.values()).T # num_points, num_components
X = vstack(norm_values.values()).T # num_points, num_components
original_indices = arange(X.shape[0])
representative_values = []
counts = []
while original_indices.size:
    # compute distance matrix
    D = amax(abs(X[:, newaxis, :]-X[newaxis, :, :]), axis=2)
    C = sum(D<epsilon, axis=1)
    i = argmax(C) # most representative point
    representative_values.append(Y[original_indices[i], :])
    k = D[:, i]>=epsilon
    original_indices = original_indices[k]
    X = X[k, :]
    counts.append(C[i])
representative_values = array(representative_values)
counts = array(counts)
print 'Num rep values', representative_values.shape[0]
print 'Representation level:', around(100.*counts/sum(counts), 1)
print 'Representation of first 10:', round(sum(counts[:10])*100.0/sum(counts), 1)
num_repval_to_show = min(representative_values.shape[0], 10)
nc = 6
figure(dpi=75, figsize=(nc*2, 1.5*num_repval_to_show))
for repval in range(num_repval_to_show):
    cur_params = {}
    for i, k in enumerate(params.keys()):
        cur_params[k] = representative_values[repval, i]
    cur_raw_res = simple_model(1, cur_params)
    cur_res = simple_model_results(1, cur_raw_res, error_func, weighted=False, interpolate_bmf=True)
    ax = subplot(num_repval_to_show, nc, nc*repval+1)
    ax.set_frame_on(False)
    xticks([])
    yticks([])
    tab_items = [(k, '%.2f' % v) for k, v in cur_params.items()]
    #tab_items = [('Representation (%)', around(100.*counts[repval]/sum(counts), 1))]+tab_items
    table(loc='center', cellLoc='center', cellText=tab_items)
    ylabel('%.2f%%' % (100.*counts[repval]/sum(counts)))
    for i, (ylab, v) in enumerate([('Extracted phase (deg)', cur_res.peak_phase.T*180/pi),
                                   ('rMTF (peak)', cur_res.norm_measures['peak'].T),
                                   ('rMTF (mean)', cur_res.norm_measures['mean'].T),
                                   ('tMTF (vs)', cur_res.norm_measures['vs'].T),
                                   ('Onsettiness', cur_res.norm_measures['onsettiness'].T),
                                   ]):
        subplot(num_repval_to_show, nc, nc*repval+2+i)
        plot(dietz_fm, v, '-o', label='Model')
        if i==0:
            errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--r', label='Data')
            grid()
            ylim(0, 180)
        else:
            ylim(0, 1)
        if repval==num_repval_to_show-1:
            xlabel('Modulation frequency (Hz)')
        if repval==0:
            title(ylab)
    
tight_layout()
