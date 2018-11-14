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

# %matplotlib inline
from brian2 import *
import ipywidgets as ipw

# +
def compare_them(tau_e_x_ms, tau_i_v_ms, fm_Hz):
    eqs = '''
    I = 0.5*(1-cos(2*pi*fm*t)) : 1
    out_onset = x-v : 1
    dv/dt = out_onset/tau_i_v : 1
    dx/dt = (I-x)/tau_e_x : 1
    out_ei = Re-Ri : 1
    dRe/dt = (I-Re)/tau_e_x : 1
    dRi/dt = (I-Ri)/tau_i_v : 1
    tau_e_x = tau_e_x_ms*ms : second
    tau_i_v = tau_i_v_ms*ms : second
    fm = fm_Hz*Hz : Hz
    '''
    G = NeuronGroup(1, eqs)
    M = StateMonitor(G, True, record=True)
    run(3/(fm_Hz*Hz))
    figure(figsize=(12, 3), dpi=75)
    subplot(131)
    plot(M.t/ms, M.v[0], label='v')
    plot(M.t/ms, M.Ri[0], '--', label='Ri')
    fill_between(M.t/ms, 0, M.I[0], color=(0.9,)*3, zorder=-2)
    legend(loc='best')
    subplot(132)
    plot(M.t/ms, M.x[0], label='x')
    plot(M.t/ms, M.Re[0], '--', label='Re')
    fill_between(M.t/ms, 0, M.I[0], color=(0.9,)*3, zorder=-2)
    legend(loc='best')
    subplot(133)
    omax = max(amax(M.out_ei[:]), amax(M.out_onset[:]))
    plot(M.t/ms, M.out_ei[0], label='EI')
    plot(M.t/ms, M.out_onset[0], label='Onset')
    fill_between(M.t/ms, 0, M.I[0]*omax, color=(0.9,)*3, zorder=-2)
    legend(loc='best')
    tight_layout()
    
w = ipw.interactive(compare_them,
        tau_e_x_ms=ipw.FloatSlider(min=0.1, max=10, step=0.1, value=0.1,
            description=r"Time constant E/x (ms)"),
        tau_i_v_ms=ipw.FloatSlider(min=0.1, max=10, step=0.1, value=1,
            description=r"Time constant I/v (ms)"),
        fm_Hz=ipw.FloatSlider(min=4, max=64, step=1, value=4,
            description=r"$f_m$ (Hz)"),
        )

# Improve layout
for child in w.children:
    if isinstance(child, ipw.ValueWidget):
        child.layout.width = '100%'
        child.style = {'description_width': '30%'}
        child.continuous_update = False

display(w)
