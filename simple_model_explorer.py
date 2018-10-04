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

# # Simple models of Dietz AMBB stimuli

# +
from brian2 import *
from model_explorer_jupyter import *
import ipywidgets as ipw
from scipy.interpolate import interp1d

def normed(X, *args):
    m = max(amax(abs(Y)) for Y in (X,)+args)
    return X/m

dietz_fm = array([4, 8, 16, 32, 64])*Hz
dietz_phase = array([37, 40, 62, 83, 115])*pi/180
dietz_phase_std = array([46, 29, 29, 31, 37])*pi/180
# -

# Basic model, showing extracted phase curve and MTF

# +
def adapted_signal_inh(fm_array, taue, taui, taua, alpha, beta, gamma, level, fc, tauihc):
    k = alpha/taua
    R = (1-alpha)/taua
    gain = 10**(level/20.)
    eqs = '''
    carrier = clip(cos(2*pi*fc*t), 0, Inf) : 1
    A_raw = (carrier*gain*0.5*(1-cos(2*pi*fm*t)))**gamma : 1
    dA_filt/dt = (A_raw-A)/(int(tauihc==0*ms)*1*second+tauihc) : 1
    A = A_raw*int(tauihc==0*ms)+A_filt*int(tauihc>0*ms) : 1
    dQ/dt = -k*Q*A+R*(1-Q) : 1
    AQ = A*Q : 1
    dAe/dt = (AQ-Ae)/taue : 1
    dAi/dt = (AQ-Ai)/taui : 1
    out = clip(Ae-beta*Ai, 0, Inf) : 1
    fm : Hz
    '''
    G = NeuronGroup(len(fm_array), eqs, method='euler', dt=0.1*ms)
    G.fm = fm_array
    M = StateMonitor(G, True, record=True)
    Network(G, M).run(1*second)
    return M.t[:], M.A[:], M.Q[:], M.Ae[:], M.Ai[:], M.out[:]

def peak_ipd_inh(fm_array, taue, taui, taua, alpha, beta, gamma, level,
                 fc, tauihc, weighted):
    T, A, Q, Ae, Ai, out = adapted_signal_inh(
        fm_array, taue, taui, taua, alpha, beta, gamma, level, fc, tauihc)
    peak_phases = []
    peak_fr = []
    mean_fr = []
    vs = []
    for outcur, fm in zip(out, fm_array):
        n = int(round(0.5*second*fm))
        I = logical_and(T>=n/fm, T<(n+1)/fm)
        TI = T[I]
        outI = outcur[I]
        phase = (2*pi*fm*TI) % (2*pi)
        if weighted:
            peak_phase = (angle(sum(outI*exp(1j*phase)))+2*pi)%(2*pi)
        else:
            peak = TI[argmax(outI)]
            peak_phase = (peak*2*pi*fm) % (2*pi)
        peak_phases.append(peak_phase)
        peak_fr.append(amax(outI))
        mean_fr.append(mean(outI))
        vs.append(abs(sum(outI*exp(1j*phase)))/sum(outI))
    return array(peak_phases), array(peak_fr), array(mean_fr), array(vs), T, A, Q, Ae, Ai, out

def show_peak_ipd_inh(taue, taui, taua, alpha, beta, gamma, level, fc, tauihc, weighted):
    fm_array = 2**linspace(log2(4), log2(256), 40)*Hz
    phi, fr, meanfr, vs, T, A, Q, Ae, Ai, out = peak_ipd_inh(
        fm_array, taue*ms, taui*ms, taua*ms, alpha, beta, gamma, level,
        fc*Hz, tauihc*ms, weighted)
    figure(figsize=(14, 10))
    gs = GridSpec(2, 6, height_ratios=[3, 2])
    subplot(gs[0, :3])
    semilogx(fm_array, phi*180/pi, '-', label='Model')
    errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--or', label='Data')
    legend(loc='upper left')
    grid()
    ylim(0, 180)
    xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')
    xticks([4, 8, 16, 32, 64, 128, 256], [4, 8, 16, 32, 64, 128, 256])
    subplot(gs[0, 3:])
    norm_fr = fr/amax(fr)
    norm_meanfr = meanfr/amax(meanfr)
    semilogx(fm_array, norm_fr, '-', label='Peak rate')
    semilogx(fm_array, norm_meanfr, '-', label='Mean rate')
    semilogx(fm_array, vs, '--', label='Vector strength')
    xticks([4, 8, 16, 32, 64, 128, 256], [4, 8, 16, 32, 64, 128, 256])
    legend(loc='best')
    xlabel('Modulation frequency (Hz)')
    ylabel('Relative MTF / Vector strength')
    ylim(0, 1)
    
    for plot_idx, fm_idx in enumerate(searchsorted(asarray(fm_array), [4, 16, 64])):
        fm = fm_array[fm_idx]
        subplot(gs[1, 2*plot_idx:2*plot_idx+2])
        n = int(round(0.5*second*fm))
        I = array(T*fm, dtype=int)==n
        fill_between((fm*T[I]*360)%360, 0, A[fm_idx][I]/(10**(level/20.)), color=(0.9,)*3, zorder=-2)
        xlabel('Phase (deg)')
        xlim(0, 360)
        ax = gca().twiny()
        plot(T[I]/ms, normed(Q[fm_idx][I]), label='Q')
        plot(T[I]/ms, normed(Q[fm_idx][I]*A[fm_idx][I]), label='A*Q')
        plot(T[I]/ms, normed(Ae[fm_idx][I], Ai[fm_idx][I]), label='Ae')
        plot(T[I]/ms, normed(Ai[fm_idx][I], Ae[fm_idx][I]), '--', label='Ai')
        plot(T[I]/ms, normed(out[fm_idx][I]), '-k', label='out')
        if fm_idx==0:
            legend(loc='lower left')
        xlabel('$f_m=%d$ Hz\nTime (ms)' % round(fm))
        xlim(n/fm/ms, (n+1)/fm/ms)
    tight_layout()
    
w = ipw.interactive(show_peak_ipd_inh,
        fc=ipw.FloatSlider(min=0, max=2000, step=100, value=0,
            description=r"Carrier frequency (0=env only) $f_c$ (Hz)"),
        tauihc=ipw.FloatSlider(min=0, max=10, step=0.1, value=0,
            description=r"Inner hair cell time constant (0=off) $\tau_{ihc}$ (ms)"),
        weighted=ipw.Checkbox(description="Use weighted instead of peak phase", value=False),
        taue=ipw.FloatSlider(min=0.1, max=10, step=0.1, value=0.1,
            description=r"Excitatory filtering time constant $\tau_e$ (ms)"),
        taui=ipw.FloatSlider(min=0.1, max=10, step=0.1, value=0.5,
            description=r"Inhibitory filtering time constant $\tau_i$ (ms)"),
        taua=ipw.FloatSlider(min=0.1, max=10, step=0.1, value=5,
            description=r"Adaptation time constant $\tau_a$ (ms)"),
        alpha=ipw.FloatSlider(min=0, max=0.99, step=0.01, value=0.8,
            description=r"Adaptation strength $\alpha$"),
        beta=ipw.FloatSlider(min=0, max=2, step=0.01, value=1.0,
            description=r"Inhibition strength $\beta$"),
        gamma=ipw.FloatSlider(min=0.1, max=1, step=0.01, value=1.0,
            description=r"Compression power $\gamma$"),
        level=ipw.FloatSlider(min=-90, max=90, step=5, value=0,
            description=r"Relative sound level $L$ (dB)"),
        )

# Improve layout
for child in w.children:
    if isinstance(child, ipw.ValueWidget):
        child.layout.width = '100%'
        child.style = {'description_width': '30%'}
        child.continuous_update = False

display(w)
