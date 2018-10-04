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
#
# TODO:
#
# * following parameters should be better for popmap: taui_ms=(3, 7), level=(0, 20), taua_ms=(0.5, 5), beta=(0, 2), alpha=(0, 0.99), taue_ms=(0.1, 1)
# * Use accumulators to save memory and allow for larger runs

# +
# %matplotlib notebook
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
from sklearn.mixture import BayesianGaussianMixture

BrianLogger.suppress_name('resolution_conflict')

def normed(X, *args):
    m = max(amax(abs(Y)) for Y in (X,)+args)
    return X/m

mem = joblib.Memory(location='.', bytes_limit=10*1024**3, verbose=0) # 10 GB max cache
# -

# Raw data we want to model

dietz_fm = array([4, 8, 16, 32, 64])*Hz
dietz_phase = array([37, 40, 62, 83, 115])*pi/180
dietz_phase_std = array([46, 29, 29, 31, 37])*pi/180

# Basic model definitions

# +
@mem.cache
def simple_model(N, params):
    min_tauihc = 0.1*ms
    eqs = '''
    carrier = clip(cos(2*pi*fc*t), 0, Inf) : 1
    A_raw = (carrier*gain*0.5*(1-cos(2*pi*fm*t)))**gamma : 1
    dA_filt/dt = (A_raw-A)/(int(tauihc<min_tauihc)*1*second+tauihc) : 1
    A = A_raw*int(tauihc<min_tauihc)+A_filt*int(tauihc>=min_tauihc) : 1
    dQ/dt = -k*Q*A+R*(1-Q) : 1
    AQ = A*Q : 1
    dAe/dt = (AQ-Ae)/taue : 1
    dAi/dt = (AQ-Ai)/taui : 1
    out = clip(Ae-beta*Ai, 0, Inf) : 1
    gain = 10**(level/20.) : 1
    R = (1-alpha)/taua : Hz
    k = alpha/taua : Hz
    fc = fc_Hz*Hz : Hz
    fc_Hz : 1
    fm : Hz
    tauihc = tauihc_ms*ms : second
    taue = taue_ms*ms : second
    taui = taui_ms*ms : second
    taua = taua_ms*ms : second
    tauihc_ms : 1
    taue_ms : 1
    taui_ms : 1
    taua_ms : 1
    alpha : 1
    beta : 1
    gamma : 1
    level : 1
    '''
    G = NeuronGroup(N, eqs, method='euler', dt=0.1*ms)
    G.set_states(params)
    G.tauihc_ms['tauihc_ms<min_tauihc/ms'] = 0
    G.Q = 1
    M = StateMonitor(G, 'out', record=True)
    net = Network(G)
    net.run(.25*second)
    net.add(M)
    net.run(.25*second)
    return M.t[:], M.out[:]

def extract_peak_phase(N, t, out, error_func, weighted, interpolate_bmf=False):
    out = reshape(out, (N, len(dietz_fm), len(t)))
    fm = dietz_fm
    n = array(around(0.25*second*fm), dtype=int)
    idx = (t[newaxis, newaxis, :]<(n/fm)[newaxis, :, newaxis])+zeros(out.shape, dtype=bool)
    out[idx] = 0
    if weighted:
        phase = (2*pi*fm[newaxis, :, newaxis]*t[newaxis, newaxis, :]) % (2*pi)
        peak_phase = (angle(sum(out*exp(1j*phase), axis=2))+2*pi)%(2*pi)
    else:
        peak = t[argmax(out, axis=2)] # shape (N, n_fm)
        peak_phase = (peak*2*pi*fm[newaxis, :]) % (2*pi) # shape (N, n_fm)
    peak_fr = amax(out, axis=2) # shape (N, n_fm)
    norm_peak_fr = peak_fr/amax(peak_fr, axis=1)[:, newaxis]
    mse = error_func(dietz_phase[newaxis, :], peak_phase) # sum over fm, mse has shape N
    mse_norm = (mse-amin(mse))/(amax(mse)-amin(mse))
    bmf = asarray(dietz_fm)[argmax(norm_peak_fr, axis=1)]
    moddepth = 1-amin(norm_peak_fr, axis=1)
    # interpolated bmf
    if interpolate_bmf:
        fm_interp = linspace(4, 64, 100)
        for cx in xrange(N):
            cur_fr = norm_peak_fr[cx, :]
            fr_interp_func = interp1d(dietz_fm, cur_fr, kind='quadratic')
            bmf[cx] = fm_interp[argmax(fr_interp_func(fm_interp))]
    return peak_phase, peak_fr, norm_peak_fr, mse, mse_norm, bmf, moddepth
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
def map2d(M, weighted, error_func_name, **kwds):
    global curfig
    # Set up ranges of variables, and generate arguments to pass to model function
    error_func = error_functions[error_func_name]
    selected_axes = ('alpha', 'beta')
    axis_ranges = dict((k, linspace(*(v+(M,)))) for k, v in kwds.items() if k in selected_axes)
    axis_ranges['fm'] = dietz_fm
    array_kwds = meshed_arguments(selected_axes+('fm',), kwds, axis_ranges)
    vx, vy = selected_axes
    shape = array_kwds[vx].shape
    N = array_kwds[vx].size
    array_kwds[vx].shape = N
    array_kwds[vy].shape = N
    array_kwds['fm'].shape = N
    # Run the model
    t, out = simple_model(N, array_kwds)
    (all_peak_phase, all_peak_fr, all_norm_peak_fr,
     mse, mse_norm, bmf, moddepth) = extract_peak_phase(M*M, t, out, error_func, weighted)    
    # Analyse the data
    peak_phase = all_peak_phase.reshape((M, M, -1))
    norm_peak_fr = all_norm_peak_fr.reshape((M, M, -1))
    bmf.shape = moddepth.shape = mse.shape = mse_norm.shape = (M, M)
    # Properties of lowest MSE value
    idx_best_y, idx_best_x = unravel_index(argmin(mse), mse.shape)
    xbest = axis_ranges[vx][idx_best_x]
    ybest = axis_ranges[vy][idx_best_y]
    best_peak_phase = peak_phase[idx_best_y, idx_best_x, :]
    best_norm_peak_fr = norm_peak_fr[idx_best_y, idx_best_x, :]
    print 'Best: {vx} = {xbest}, {vy} = {ybest}'.format(vx=vx, vy=vy, xbest=xbest, ybest=ybest)
    # Plot the data
    extent = (kwds[vx]+kwds[vy])
    def labelit(titletext):
        plot([xbest], [ybest], '+w')
        title(titletext)
        xlabel(r'Adaptation strength $\alpha$')
        ylabel(r'Inhibition strength $\beta$')
        cb = colorbar()
        cb.set_label(titletext, rotation=270, labelpad=20)

    #mse = zoom(mse, 100./M, order=1)
    mse_deg = mse*180/pi
    imshow(mse_deg, origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, extent=extent)
    labelit(error_func_name)
    cs = contour(mse_deg, origin='lower', aspect='auto',
                 levels=[15, 30, 45], colors='w',
                 extent=extent)
    clabel(cs, colors='w', inline=True, fmt='%d')

    
population_summary_methods = {
    'Mean': mean,
    'Best': amin,
    }

def popmap(M, num_params, blur_width, error_cutoff_deg,
           weighted, error_func_name,
           pop_summary_name='Best', smoothing=True,
           loc_summary=121, loc_close=122,
           **kwds):
    global curfig
    # always use the same random seed for cacheing
    seed(34032483)    
    # Set up ranges of variables, and generate arguments to pass to model function
    selected_axes = ('alpha', 'beta')
    pop_summary = population_summary_methods[pop_summary_name]
    error_func = error_functions[error_func_name]
    axis_ranges = dict((k, linspace(*(v+(M,)))) for k, v in kwds.items() if k in selected_axes)
    axis_ranges['fm'] = dietz_fm
    axis_ranges['temp'] = zeros(num_params)
    array_kwds = meshed_arguments(selected_axes+('temp', 'fm'), kwds, axis_ranges)
    del array_kwds['temp']
    vx, vy = selected_axes
    shape = array_kwds[vx].shape # shape will be (M, M, num_params, len(dietz_fm))
    N = array_kwds[vx].size
    for k, (low, high) in kwds.items():
        if k not in selected_axes:
            array_kwds[k] = rand(N)*(high-low)+low
        array_kwds[k].shape = N
    array_kwds['fm'].shape = N
    # Run the model
    t, out = simple_model(N, array_kwds)
    (all_peak_phase, all_peak_fr, all_norm_peak_fr,
     mse, mse_norm, bmf, moddepth) = extract_peak_phase(M*M*num_params, t, out,
                                                        error_func, weighted)
    # Analyse the data
    peak_phase = all_peak_phase.reshape((M, M, num_params, -1))
    norm_peak_fr = all_norm_peak_fr.reshape((M, M, num_params, -1))
    bmf.shape = moddepth.shape = mse.shape = mse_norm.shape = (M, M, num_params)
    mse = mse*180/pi
    mse_summary = pop_summary(mse, axis=2)
    mse_close = 1.0*sum(mse<error_cutoff_deg, axis=2)/num_params
    # Plot the data
    if smoothing:
        mse_summary = gaussian_filter(mse_summary, blur_width*M, mode='nearest')
        mse_summary = zoom(mse_summary, 100./M, order=1)
        mse_close = gaussian_filter(mse_close, blur_width*M, mode='nearest')
        mse_close = zoom(mse_close, 100./M, order=1)
    extent = (kwds[vx]+kwds[vy])
    
    subplot(loc_summary)
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

    subplot(loc_close)
    imshow(100.*mse_close, origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, extent=extent)
    xlabel(r'Adaptation strength $\alpha$')
    ylabel(r'Inhibition strength $\beta$')
    cb = colorbar()
    cb.set_label("Percent within cutoff", rotation=270, labelpad=20)
    

def parameter_space(N, M, M_popmap,
                    num_params, blur_width, popmap_error_cutoff,
                    weighted,
                    error_func_name, error_cutoffs,
                    num_examples, example_error_cutoff,
                    search_params, adapt_params, inhib_params, popmap_params,
                    ):
    # always use the same random seed for cacheing
    seed(34032483)
    # Get simple parameters
    error_func = error_functions[error_func_name]
    kwds = search_params
    # Set up array keywords
    array_kwds = {}
    param_values = {}
    for k, (low, high) in kwds.items():
        v = rand(N)*(high-low)+low
        param_values[k] = v
        fm, v = meshgrid(dietz_fm, v) # fm and v have shape (N, len(dietz_fm))!
        fm.shape = fm.size
        v.shape = v.size
        array_kwds['fm'] = fm
        array_kwds[k] = v
    # Run the model
    t, out = simple_model(N*len(dietz_fm), array_kwds)
    out3d = reshape(out, (N, len(dietz_fm), len(t)))
    (peak_phase, peak_fr, norm_peak_fr,
     mse, mse_norm, bmf, moddepth) = extract_peak_phase(N, t, out, error_func, weighted)
    # Properties of lowest MSE value
    idx_best = argmin(mse)
    best_peak_phase = peak_phase[idx_best, :]
    best_norm_peak_fr = norm_peak_fr[idx_best, :]
    bestvals = []
    for k in kwds.keys():
        v = param_values[k][idx_best]
        bestvals.append('%s=%.2f' % (k, v))
    print 'Best: ' + ', '.join(bestvals)
    # Properties of all data below error cutoff
    varying_param_values = {}
    param_value_index = {}
    for j, (k, v) in enumerate(param_values.items()):
        param_value_index[k] = j
        if amin(v)!=amax(v):
            varying_param_values[k] = v    
    all_params = vstack(param_values.values()).T
    keep_indices = {}
    keep_params = {}
    for error_cutoff in error_cutoffs:
        KI = keep_indices[error_cutoff] = (mse<error_cutoff*pi/180).nonzero()[0]
        KP = keep_params[error_cutoff] = all_params[KI, :] # (paramset, param)
    # All data below lowest error cutoff, we find distant points by clustering
    model = BayesianGaussianMixture(n_components=num_examples)
    KP = keep_params[example_error_cutoff]
    model.fit(KP)
    example_indices = []
    for i in range(num_examples):
        # find closest index to cluster mean i
        j = argmin(sum((model.means_[i, :][newaxis, :]-KP)**2, axis=1))
        example_indices.append(keep_indices[example_error_cutoff][j])
        s = 'Example %d (weight %.2f):\n\t' % (i+1, model.weights_[i])
        cpts = []
        for k in varying_param_values.keys():
            c = KP[j, param_value_index[k]]
            if c:
                cpts.append('%s = %.3f' % (k, c))
        s += ',\n\t'.join(cpts)
        print s        
    ############# Plot the data
    curfig = figure(dpi=65, figsize=(14, 16))
    clf()
    height_ratios = [1, 1, 0.5, 0.5, 0.6]
    gs = GridSpec(5, 12, height_ratios=height_ratios)
    
    # Plot the extracted phase curves
    subplot(gs[0, :6])
    transp = clip(0.3*100./N, 0.01, 1)
    plot(dietz_fm/Hz, peak_phase.T*180/pi, '-', color=(0.4, 0.7, 0.4, transp), label='Model (all)')
    plot(dietz_fm/Hz, best_peak_phase*180/pi, '-ko', lw=2, label='Model (best)')
    errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--r', label='Data')
    handles, labels = gca().get_legend_handles_labels()
    lab2hand = OrderedDict()
    for h, l in zip(handles, labels):
        lab2hand[l] = h
    legend(lab2hand.values(), lab2hand.keys(), loc='upper left')
    grid()
    ylim(0, 180)
    xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')
    
    # Plot the MTFs
    subplot(gs[0, 6:])
    lines = plot(dietz_fm/Hz, norm_peak_fr.T, '-')
    for i, line in enumerate(lines):
        line.set_color(cm.YlGnBu_r(mse_norm[i], alpha=transp))
    lines[argmin(mse)].set_alpha(1)
    lines[argmax(mse)].set_alpha(1)
    lines[argmin(mse)].set_label('Model (all, best MSE)')
    lines[argmax(mse)].set_label('Model (all, worst MSE)')
    plot(dietz_fm/Hz, best_norm_peak_fr, '-ko', lw=2)
    fm_interp = linspace(4, 64, 1000)
    fr_interp_func = interp1d(dietz_fm/Hz, best_norm_peak_fr, kind='quadratic')
    plot(fm_interp, fr_interp_func(fm_interp), ':k', lw=2)
    legend(loc='best')
    ylim(0, 1)
    xlabel('Modulation frequency (Hz)')
    ylabel('Relative MTF')
    
    # Plot the examples
    n = array(around(0.25*second*dietz_fm), dtype=int)
    for i, j in enumerate([0, -1]):
        fm = dietz_fm[j]
        cur_n = n[j]
        idx = logical_and(t>=(cur_n/fm), t<=((cur_n+1)/fm))
        cur_t = t[idx]
        phase = (2*pi*fm*cur_t)%(2*pi)
        env = 0.5*(1-cos(phase))
        subplot(gs[1, 4*i:4*(i+1)])
        if i==0:
            title('Low frequency (4 Hz)\n\n')
        else:
            title('High frequency (64 Hz)\n\n')
        fill_between(phase*180/pi, 0, env, color=(0.9,)*3, zorder=-2)
        ylim(0, 1.1)
        xlabel('Phase (deg)')
        xlim(0, 360)
        ax = gca().twiny()
        for ei_idx, example_index in enumerate(example_indices):
            plot((cur_t-amin(cur_t))/ms, normed(out3d[example_index, j, idx]),
                 '-', c='C%d'%ei_idx, lw=2)
        xlabel('Time (ms)')
        xlim(0, 1/fm/ms)
        ylim(0, 1.1)
        
    subplot(gs[1, 8:12])
    for i, example_index in enumerate(example_indices):
        plot(dietz_fm/Hz, peak_phase[example_index, :]*180/pi,
             '-o', lw=2, c='C%d' % (i),
             label='Model %d' % (i+1))
    errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--r', label='Data')
    legend(loc='best')
    grid()
    ylim(0, 180)
    xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')    
    
    # Plot the histograms
    for i, param_name in enumerate(sorted(varying_param_values.keys())):
        subplot(gs[2+i//4, (i%4)*3:(i%4+1)*3])
        xlabel(latex_parameter_names[param_name])
        yticks([])
        for j, error_cutoff in enumerate(error_cutoffs[::-1]):
            hist(keep_params[error_cutoff][:, param_value_index[param_name]],
                 bins=20, range=kwds[param_name], histtype='stepfilled',
                 fc=(1-0.7*(j+1)/len(error_cutoffs),)*3,
                 label="Error<%d deg" % error_cutoff)
        for k, j in enumerate(example_indices):
            v = all_params[j, param_value_index[param_name]]
            axvline(v, ls='--', c='C%d'%(k), lw=2)
            
    #legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    legend(loc='best') # TODO: better location
    
    # Plot 2D maps
    subplot(gs[4, 0:3])
    map2d(M, weighted, error_func_name, **adapt_params)
    title('Fits close to best pure adaptation')
    
    subplot(gs[4, 3:6])
    map2d(M, weighted, error_func_name, **inhib_params)
    title('Fits close to best pure inhbition')
    
    popmap(M=M_popmap, num_params=num_params, blur_width=blur_width,
           weighted=weighted, error_func_name=error_func_name,
           error_cutoff_deg=popmap_error_cutoff,
           smoothing=True, loc_summary=gs[4, 6:9], loc_close=gs[4, 9:12],
           **popmap_params)
    
    tight_layout()

    # Label panels
    _, offsets, _, _ = gs.get_grid_positions(gcf())
    for i, (o, s) in enumerate(zip(offsets, 'ABC D')):
        text(0, o, s, transform=gcf().transFigure, fontsize=16)

        
#N = 1000; M=20; M_popmap=10; num_params=20; blur_width=0.2 # quick, low quality
N = 10000; M=40; M_popmap=20; num_params=100; blur_width=0.05 # medium quality

parameter_space(N=N, M=M, M_popmap=M_popmap, num_params=num_params, blur_width=blur_width,
                popmap_error_cutoff=30,
                weighted=False, error_func_name="Max error",
                num_examples=3, example_error_cutoff=30,
                error_cutoffs=[15, 30, 45],
                search_params=dict(
                    taui_ms=(0.1, 10), taue_ms=(0.1, 10), taua_ms=(0.1, 10),
                    level=(-25, 25), alpha=(0, 0.99), beta=(0, 2),
                    gamma=(0.1, 1)),
                adapt_params=dict(
                    alpha=(0, 0.99), beta=(0, 2),
                    taui_ms=8.39, level=9.27, taua_ms=1.90, taue_ms=1.22, gamma=0.78,
                    ),
                inhib_params=dict(
                    alpha=(0, 0.99), beta=(0, 2),
                    taui_ms=2.22, level=4.49, taua_ms=8.94, taue_ms=0.16, gamma=0.70,
                    ),
                popmap_params=dict(
                    taui_ms=(1, 10), level=(0, 15), taua_ms=(1, 10),
                    alpha=(0, 0.99), beta=(0, 2),
                    taue_ms=(0.1, 1.5), gamma=(0.6, 0.9),
                    ),
               )
savefig('figure_parameter_space.pdf')
