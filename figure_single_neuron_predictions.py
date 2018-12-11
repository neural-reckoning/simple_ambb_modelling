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
