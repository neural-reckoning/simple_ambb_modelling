# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Figure: Basic mechanisms

# +
# %matplotlib notebook
import warnings
warnings.filterwarnings("ignore")
from brian2 import *
from simple_model import *

def normed(X, *args):
    m = max(amax(abs(Y)) for Y in (X,)+args)
    return X/m


# -

# First row: pure differentiation

# +
def pure_differentiation(only_this=False):
    h = 1 if only_this else 4
    if only_this:
        figure(dpi=75, figsize=(10, 3.5))
    extracted_phase = []
    for i, fm in enumerate(dietz_fm):
        t = linspace(0*ms, 1/fm, 100)
        phase = linspace(0, 2*pi, 100)
        env = 0.5*(1-cos(2*pi*fm*t))
        diff_env = fm*pi*sin(2*pi*fm*t)
        clipped_diff_env = clip(diff_env, 0*Hz, Inf*Hz)
        extracted_phase.append(phase[argmax(clipped_diff_env)])
        if i>0 and i<len(dietz_fm)-1:
            continue
        if i==0:
            subplot(h, 3, 1)
            title('Low frequency (4 Hz)\n\n')
        else:
            subplot(h, 3, 2)
            title('High frequency (64 Hz)\n\n')
        fill_between(phase*180/pi, 0, env, color=(0.9,)*3, zorder=-2)
        ylim(0, 1.1)
        if only_this:
            xlabel('Phase (deg)')
        if i==0:
            ylabel('Differentiation\n', fontsize=14)
        xlim(0, 360)
        ax = gca().twiny()
        plot(t/ms, normed(clipped_diff_env), '-k', lw=2)
        xlabel('Time (ms)')
        xlim(0, 1/fm/ms)
        ylim(0, 1.1)
    subplot(h, 3, 3)
    extracted_phase = array(extracted_phase)
    errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--or', label='Data')
    plot(dietz_fm/Hz, extracted_phase*180/pi, '-ok', lw=2, label='Model')
    legend(loc='upper left')
    ylim(0, 180)
    xlim(0, 70)
    xticks(dietz_fm/Hz)
    if only_this:
        xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')
    title('All frequencies\n\n')
    tight_layout()

pure_differentiation(True)


# +
def with_model(name, row, only_this=False, **params):
    h = 1 if only_this else 4
    if only_this:
        figure(dpi=75, figsize=(10, 3.5))
        row = 1
    res = simple_model(1, params, record=['out'])
    t = res.t
    out = reshape(res.out, (len(dietz_fm), len(t)))
    n = array(around(0.25*second*dietz_fm), dtype=int)
    idx = t[newaxis, :]<(n/dietz_fm)[:, newaxis]
    out[idx] = 0
    peak = t[argmax(out, axis=1)]
    extracted_phase = (peak*2*pi*dietz_fm) % (2*pi)
    for i, j in enumerate([0, -1]):
        fm = dietz_fm[j]
        cur_n = n[j]
        idx = logical_and(t>=(cur_n/fm), t<=((cur_n+1)/fm))
        cur_t = t[idx]
        phase = (2*pi*fm*cur_t)%(2*pi)
        env = 0.5*(1-cos(phase))
        if i==0:
            subplot(h, 3, (row-1)*3+1)
            if only_this:
                title('Low frequency (4 Hz)\n\n')
        else:
            subplot(h, 3, (row-1)*3+2)
            if only_this:
                title('High frequency (64 Hz)\n\n')
        if i==0:
            ylabel(name+'\n', fontsize=14)
        fill_between(phase*180/pi, 0, env, color=(0.9,)*3, zorder=-2)
        ylim(0, 1.1)
        if only_this or row==4:
            xlabel('Phase (deg)')
        xlim(0, 360)
        ax = gca().twiny()
        plot((cur_t-amin(cur_t))/ms, normed(out[j, idx]), '-k', lw=2)
        if only_this:
            xlabel('Time (ms)')
        xlim(0, 1/fm/ms)
        ylim(0, 1.1)
    subplot(h, 3, (row-1)*3+3)
    errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--or', label='Data')
    plot(dietz_fm/Hz, extracted_phase*180/pi, '-ok', lw=2, label='Model')
    if only_this:
        legend(loc='upper left')
    ylim(0, 180)
    xlim(0, 70)
    xticks(dietz_fm/Hz)
    if only_this:
        xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')
    tight_layout()

# Onset only
# with_model("Onset", 1, True,
#            taui_ms=1.56, fc_Hz=0.00, level=0.00, tauihc_ms=0.00,
#            taua_ms=6.69, beta=1.06, alpha=0.00, taue_ms=0.63, gamma=1.00)
# Adaptation only
# with_model("Adaptation", 1, True,
#            taui_ms=8.23, fc_Hz=0.00, level=0.00, tauihc_ms=0.00,
#            taua_ms=0.79, beta=0.00, alpha=0.99, taue_ms=1.77, gamma=1.00)
# Complex model
with_model("Complex model", 1, True,
           taui_ms=7.51, fc_Hz=0.00, level=18.73, tauihc_ms=0.00,
           taua_ms=3.78, beta=0.41, alpha=0.72, taue_ms=1.38, gamma=0.84)
# -

# Second to fourth row: use model

# Full figure:

# +
figure(dpi=75, figsize=(10, 10))
pure_differentiation()
# Onset only
with_model("Onset", 2, False,
           taui_ms=1.56, fc_Hz=0.00, level=0.00, tauihc_ms=0.00,
           taua_ms=6.69, beta=1.06, alpha=0.00, taue_ms=0.63, gamma=1.00)
# Adaptation only
with_model("Adaptation", 3, False,
           taui_ms=8.23, fc_Hz=0.00, level=0.00, tauihc_ms=0.00,
           taua_ms=0.79, beta=0.00, alpha=0.99, taue_ms=1.77, gamma=1.00)
# Complex model
with_model("Complex model", 4, False,
           taui_ms=7.51, fc_Hz=0.00, level=18.73, tauihc_ms=0.00,
           taua_ms=3.78, beta=0.41, alpha=0.72, taue_ms=1.38, gamma=0.84)
# Annotations
for i, c in enumerate('ABCD'):
    text(0.02, 0.98-0.96*i/4, c, fontsize=22, transform=gcf().transFigure,
         horizontalalignment='left', verticalalignment='top')

savefig('figure_basic_mechanism.pdf')
# -


