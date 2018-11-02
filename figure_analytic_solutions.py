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

# %matplotlib inline
from brian2 import *
from simple_model import *
from scipy.optimize import curve_fit

# Optimisation of analytic solution of onset only model
def onsetphi(fm, beta, taue, taui):
    sigmae = 2*pi*fm*taue
    sigmai = 2*pi*fm*taui
    return arctan2(beta*sigmai+beta*sigmae**2*sigmai-sigmae*(1+sigmai**2),
                   beta-1+beta*sigmae**2-sigmai**2)
f = lambda fm, beta, taue, taui: onsetphi(fm*Hz, beta, taue*ms, taui*ms) # exact
popt, _ = curve_fit(f, dietz_fm, dietz_phase,
                       p0=(1.2, 0.1, 2.0),
                       bounds=([0, 0, 0], [inf, inf, inf]),
                      )
print popt
f_simple = lambda fm, A: arctan(A*fm) # approximate
popt_simple, _ = curve_fit(f_simple, dietz_fm, dietz_phase)
print popt_simple
errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--or', label='Data')
plot(dietz_fm/Hz, (180/pi)*f(dietz_fm/Hz, *popt), '-k', label='Onset model')
plot(dietz_fm/Hz, (180/pi)*f_simple(dietz_fm/Hz, *popt_simple), '--k', label='Simplified onset model')
xticks(dietz_fm/Hz)
ylim(0, 180)
ylabel('Phase (deg)')
xlabel(r'Modulation frequency $f_m$')
legend(loc='upper left')
savefig('figure_onset_analytic.pdf')


