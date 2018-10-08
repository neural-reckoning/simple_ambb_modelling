# ---
# jupyter:
#   celltoolbar: Initialization Cell
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

# # Maps of simple model
#
# TODO:
# * For param maps on pop/map plot, instead of showing just best param, all shown variance
# * **Sensitivity analysis** somehow
# * sampling/smoothing on population/map plot

# ## Common code / data

# + {"init_cell": true}
# %matplotlib notebook
from brian2 import *
from model_explorer_jupyter import *
import ipywidgets as ipw
from collections import OrderedDict
from scipy.interpolate import interp1d
from matplotlib import cm
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, SpectralEmbedding, MDS
from sklearn.decomposition import PCA
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from simple_model import *

def normed(X, *args):
    m = max(amax(abs(Y)) for Y in (X,)+args)
    return X/m

progress_slider, update_progress = brian2_progress_reporter()

# + {"cell_type": "markdown", "heading_collapsed": true}
# ## Definition of basic model

# + {"hidden": true, "heading_collapsed": true, "cell_type": "markdown"}
# ### Specifications of parameters

# + {"init_cell": true, "hidden": true}
parameter_specs = [
    dict(name='fc_Hz',
         description=r"Carrier frequency (0=env only) $f_c$ (Hz)",
         min=0, max=2000, step=100, value=0),
    dict(name='tauihc_ms',
         description=r"Inner hair cell time constant (<0.1=off) $\tau_{ihc}$ (ms)",
         min=0, max=10, step=0.1, value=0),
    dict(name='taue_ms',
         description=r"Excitatory filtering time constant $\tau_e$ (ms)",
         min=0.1, max=10, step=0.1, value=0.1),
    dict(name='taui_ms',
         description=r"Inhibitory filtering time constant $\tau_i$ (ms)",
         min=0.1, max=10, step=0.1, value=0.5),
    dict(name='taua_ms',
         description=r"Adaptation time constant $\tau_a$ (ms)",
         min=0.1, max=10, step=0.1, value=5),
    dict(name='alpha',
         description=r"Adaptation strength $\alpha$",
         min=0, max=0.99, step=0.01, value=0.8),
    dict(name='beta',
         description=r"Inhibition strength $\beta$",
         min=0, max=2, step=0.01, value=1.0),
    dict(name='gamma',
         description=r"Compression power $\gamma$",
         min=0.1, max=1, step=0.01, value=1.0),
    dict(name='level',
         description=r"Relative sound level $L$ (dB)",
         min=-90, max=90, step=5, value=0),
    ]

# + {"hidden": true, "heading_collapsed": true, "cell_type": "markdown"}
# ### Definition of error functions

# + {"init_cell": true, "hidden": true}
def rmse(x, y, axis=1):
    return sqrt(mean((x-y)**2, axis=axis))

def maxnorm(x, y, axis=1):
    return amax(abs(x-y), axis=axis)

error_functions = {
    'RMS error': rmse,
    'Max error': maxnorm,
    }

# + {"hidden": true, "heading_collapsed": true, "cell_type": "markdown"}
# ### Definition of dimensionality reduction methods

# + {"init_cell": true, "hidden": true}
dimensionality_reduction_methods = {
    'None': None,
    't-SNE': TSNE(n_components=2),
    'PCA': PCA(n_components=2),
    'Isomap': Isomap(n_components=2),
    'Locally linear embedding': LocallyLinearEmbedding(n_components=2),
    'Spectral embedding': SpectralEmbedding(n_components=2),
    'Multidimensional scaling': MDS(n_components=2),
    }
# -

# ## Plot types

# + {"cell_type": "markdown", "heading_collapsed": true}
# ### 2D map

# + {"init_cell": true, "hidden": true}
def plot_map2d_mse_mtf(selected_axes, **kwds):
    global curfig
    # Set up ranges of variables, and generate arguments to pass to model function
    error_func_name = kwds.pop('error_func')
    error_func = error_functions[error_func_name]
    interpolate_bmf = kwds.pop('interpolate_bmf')
    detail_settings = dict(Low=10, Medium=40, High=100)
    M = detail_settings[kwds.pop('detail')]
    weighted = kwds.pop('weighted')
    vx, vy = selected_axes
    n_fm = len(dietz_fm)
    axis_ranges = dict((k, linspace(*(v+(M,)))) for k, v in kwds.items() if k in selected_axes)
    array_kwds = meshed_arguments(selected_axes, kwds, axis_ranges)
    # Run the model
    res = simple_model(M*M, array_kwds, update_progress=update_progress)
    res = simple_model_results(M*M, res, error_func, weighted, interpolate_bmf=interpolate_bmf, shape=(M, M))
    mse = res.mse
    vs = res.raw_measures['vs']
    # Properties of lowest MSE value
    idx_best_y, idx_best_x = unravel_index(argmin(mse), mse.shape)
    xbest = axis_ranges[vx][idx_best_x]
    ybest = axis_ranges[vy][idx_best_y]
    best_peak_phase = res.peak_phase[idx_best_y, idx_best_x, :]
    best_measures = {}
    for mname, mval in res.norm_measures.items():
        best_measures[mname] = mval[idx_best_y, idx_best_x, :]
    print 'Best: {vx} = {xbest}, {vy} = {ybest}'.format(vx=vx, vy=vy, xbest=xbest, ybest=ybest)
    # Plot the data
    extent = (kwds[vx]+kwds[vy])
    def labelit(titletext):
        plot([xbest], [ybest], '+w')
        title(titletext)
        xlabel(sliders[vx].description)
        ylabel(sliders[vy].description)
        cb = colorbar()
        cb.set_label(titletext, rotation=270, labelpad=20)

    curfig = figure(dpi=48, figsize=(19, 7.5))
    clf()
    gs = GridSpec(3, 7)

    subplot(gs[0:2, :2])
    mse_deg = mse*180/pi
    imshow(mse_deg, origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, extent=extent)
    labelit(error_func_name)
    cs = contour(mse_deg, origin='lower', aspect='auto',
                 levels=[15, 30, 45], colors='w',
                 extent=extent)
    clabel(cs, colors='w', inline=True, fmt='%d')

    for oy, (pname, pdict, vsname, vsfunc) in enumerate([('BMF', res.bmf, 'Min VS', amin),
                                                         ('Modulation depth', res.moddepth, 'Max VS', amax)]):
        for ox, mname in enumerate(['peak', 'mean', 'vs', 'onsettiness', vsname]):
            if mname!=vsname:
                mval = pdict[mname]
            else:
                mval = vsfunc(vs, axis=2)
            subplot(gs[oy, 2+ox])
            vmax = 1
            if pname=='BMF' and mname!=vsname:
                vmax = 64
            imshow(mval, origin='lower left', aspect='auto',
                   interpolation='nearest', vmin=0, vmax=vmax,
                   extent=extent)
            if ox<4:
                labelit('%s (%s)' % (pname, mname))
            else:
                labelit(vsname)

    subplot(gs[2, :2])
    plot(dietz_fm/Hz, reshape(res.peak_phase, (-1, n_fm)).T*180/pi, '-', color=(0.2, 0.7, 0.2, 0.2), label='Model (all)')
    plot(dietz_fm/Hz, best_peak_phase*180/pi, '-o', lw=2, label='Model (best)')
    errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--or', label='Data')
    handles, labels = gca().get_legend_handles_labels()
    lab2hand = OrderedDict()
    for h, l in zip(handles, labels):
        lab2hand[l] = h
    legend(lab2hand.values(), lab2hand.keys(), loc='upper left')
    grid()
    ylim(0, 180)
    xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')

    for ox, mname in enumerate(['peak', 'mean', 'vs', 'onsettiness']):
        subplot(gs[2, 2+ox])
        plot(dietz_fm/Hz, reshape(res.norm_measures[mname], (M*M, n_fm)).T, '-', color=(0.2, 0.7, 0.2, 0.2))
        plot(dietz_fm/Hz, best_measures[mname], '-o')
        fm_interp = linspace(4, 64, 100)
        fr_interp_func = interp1d(dietz_fm/Hz, best_measures[mname], kind='quadratic')
        plot(fm_interp, fr_interp_func(fm_interp), ':k')
        ylim(0, 1)
        xlabel('Modulation frequency (Hz)')
        ylabel('Relative MTF')
    
    subplot(gs[2, -1])
    imshow(mean(res.norm_measures['onsettiness'], axis=2), origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, vmax=1, extent=extent)
    labelit('Onsettiness')

    tight_layout()

# + {"cell_type": "markdown", "heading_collapsed": true}
# ### Population space

# + {"init_cell": true, "hidden": true}
current_population_space_variables = {}

def plot_population_space(**kwds):
    # always use the same random seed for cacheing
    seed(34032483)
    # Get simple parameters
    maxshow = 1000
    detail_settings = dict(Low=100, Medium=1000, High=10000)
    N = detail_settings[kwds.pop('detail')]
    weighted = kwds.pop('weighted')
    error_func_name = kwds.pop('error_func')
    error_func = error_functions[error_func_name]
    interpolate_bmf = kwds.pop('interpolate_bmf')
    # Set up array keywords
    varying_params = set(k for k, (low, high) in kwds.items() if low!=high)
    # Run the model
    res = simple_model(N, kwds, update_progress=update_progress)
    res = simple_model_results(N, res, error_func, weighted=weighted, interpolate_bmf=interpolate_bmf)
    mse = res.mse
    peak_phase = res.peak_phase
    param_values = res.raw.params
    # Properties of lowest MSE value
    idx_best = argmin(mse)
    best_peak_phase = peak_phase[idx_best, :]
    best_measures = {}
    for mname, mval in res.norm_measures.items():
        best_measures[mname] = mval[idx_best, :]    
    bestvals = []
    for k in kwds.keys():
        v = param_values[k][idx_best]
        bestvals.append('%s=%.2f' % (k, v))
    print 'Best: ' + ', '.join(bestvals)
    # Properties of all data below error cutoff
    error_cutoffs = [15, 30, 45]
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
    # Computed histograms
    computed_histograms = {}
    computed_histogram_names = []
    num_histograms = 0
    for pname, pdict in [('BMF', res.bmf), ('MD', res.moddepth)]:
        for ptype in ['peak', 'mean', 'vs', 'onsettiness']:
            num_histograms += 1
            hname = '%s (%s)' % (pname, ptype)
            computed_histogram_names.append(hname)
            for error_cutoff in error_cutoffs:
                computed_histograms[hname, error_cutoff] = pdict[ptype][keep_indices[error_cutoff]]
    num_histograms += 3
    computed_histogram_names.extend(['Min VS', 'Max VS', 'Onsettiness'])
    for error_cutoff in error_cutoffs:
        KI = keep_indices[error_cutoff]
        minvs = amin(res.raw_measures['vs'], axis=1)[KI]
        maxvs = amax(res.raw_measures['vs'], axis=1)[KI]
        mean_onsettiness = mean(res.raw_measures['onsettiness'], axis=1)[KI]
        computed_histograms['Min VS', error_cutoff] = minvs
        computed_histograms['Max VS', error_cutoff] = maxvs
        computed_histograms['Onsettiness', error_cutoff] = mean_onsettiness
    num_param_histogram_rows = int(ceil(len(varying_params)/5.))
    num_computed_histogram_rows = int(ceil(num_histograms/5.))
    num_histogram_rows = num_param_histogram_rows+num_computed_histogram_rows
    # Plot the data
    curfig = figure(dpi=65, figsize=(14, 4+1.5*num_histogram_rows))
    gs = GridSpec(1+num_histogram_rows, 5, height_ratios=[2]+[1]*num_histogram_rows)
    subplot(gs[0, 0])
    transp = clip(0.3*100./N, 0.01, 1)
    plot(dietz_fm/Hz, peak_phase[:maxshow, :].T*180/pi, '-', color=(0.4, 0.7, 0.4, transp), label='Model (all)')
    plot(dietz_fm/Hz, best_peak_phase*180/pi, '-ko', lw=2, label='Model (best)')
    errorbar(dietz_fm/Hz, dietz_phase*180/pi, yerr=dietz_phase_std*180/pi, fmt='--or', label='Data')
    handles, labels = gca().get_legend_handles_labels()
    lab2hand = OrderedDict()
    for h, l in zip(handles, labels):
        lab2hand[l] = h
    legend(lab2hand.values(), lab2hand.keys(), loc='upper left')
    grid()
    ylim(0, 180)
    xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')

    for ox, mname in enumerate(['peak', 'mean', 'vs', 'onsettiness']):
        mval = res.norm_measures[mname]
        bestmval = best_measures[mname]
        subplot(gs[0, 1+ox])
        lines = plot(dietz_fm/Hz, mval[:maxshow, :].T, '-')
        for i, line in enumerate(lines):
            line.set_color(cm.YlGnBu_r(res.mse_norm[i], alpha=transp))
        lines[argmin(mse[:maxshow])].set_alpha(1)
        lines[argmax(mse[:maxshow])].set_alpha(1)
        lines[argmin(mse[:maxshow])].set_label('Model (all, best MSE)')
        lines[argmax(mse[:maxshow])].set_label('Model (all, worst MSE)')
        plot(dietz_fm/Hz, bestmval, '-ko', lw=2)
        legend(loc='best')
        ylim(0, 1)
        xlabel('Modulation frequency (Hz)')
        ylabel('MTF (%s)' % mname)
        title(mname)
    
    # Plot histograms of param values
    for i, param_name in enumerate(sorted(varying_param_values.keys())):
        subplot(gs[1+i//5, i%5])
        xlabel(param_name)
        yticks([])
        for j, error_cutoff in enumerate(error_cutoffs[::-1]):
            hist(keep_params[error_cutoff][:, param_value_index[param_name]],
                 bins=20, range=kwds[param_name], histtype='stepfilled',
                 fc=(1-0.7*(j+1)/len(error_cutoffs),)*3,
                 label="Error<%d deg" % error_cutoff)
    #legend(loc='best') # TODO: better location

    # Plot histograms of computed values
    for i, hname in enumerate(computed_histogram_names):
        subplot(gs[1+num_param_histogram_rows+i//5, i%5])
        xlabel(hname)
        yticks([])
        if hname.startswith('BMF'):
            rng = (4, 64)
        else:
            rng = (0, 1)
        for j, error_cutoff in enumerate(error_cutoffs[::-1]):
            hist(computed_histograms[hname, error_cutoff],
                 bins=20, range=rng, histtype='stepfilled',
                 fc=(1-0.7*(j+1)/len(error_cutoffs),)*3,
                 label="Error<%d deg" % error_cutoff)
    
    tight_layout()

# + {"cell_type": "markdown", "heading_collapsed": true}
# ### Combined population / 2D map

# + {"init_cell": true, "hidden": true}
def amin_from_to(arr_from, arr_to):
    i, j = mgrid[:arr_from.shape[0], :arr_from.shape[1]]
    k = argmin(arr_from, axis=2)
    return arr_to[i, j, k]

population_summary_methods = {
    'Mean': lambda arr_from, arr_to: mean(arr_to, axis=2),
    'Best': amin_from_to,
    }

def plot_population_map(selected_axes, **kwds):
    global curfig
    # always use the same random seed for cacheing
    seed(34032483)    
    # Set up ranges of variables, and generate arguments to pass to model function
    pop_summary_name = kwds.pop('pop_summary')
    pop_summary = population_summary_methods[pop_summary_name]
    error_func_name = kwds.pop('error_func')
    error_func = error_functions[error_func_name]
    error_cutoff_deg = kwds.pop('error_cutoff_deg')
    interpolate_bmf = kwds.pop('interpolate_bmf')
    detail_settings = dict(Low=(10, 20, 0.05),
                           Medium=(20, 100, 0.025),
                           High=(30, 500, 0.01))
    M, num_params, blur_width = detail_settings[kwds.pop('detail')]
    weighted = kwds.pop('weighted')
    smoothing = kwds.pop('smoothing')
    axis_ranges = dict((k, linspace(*(v+(M,)))) for k, v in kwds.items() if k in selected_axes)
    axis_ranges['temp'] = zeros(num_params)
    array_kwds = meshed_arguments(selected_axes+('temp',), kwds, axis_ranges)
    del array_kwds['temp']
    # Run the model
    res = simple_model(M*M*num_params, array_kwds, update_progress=update_progress)
    res = simple_model_results(M*M*num_params, res, error_func, weighted, shape=(M, M, num_params))
    mse = res.mse
    vx, vy = selected_axes
    # Analyse the data
    vs = res.raw_measures['vs']    
    mse = mse*180/pi
    mse_summary = pop_summary(mse, mse)
    mse_close = 1.0*sum(mse<error_cutoff_deg, axis=2)/num_params
    summary_measures = OrderedDict()
    for dname, d in [('bmf', res.bmf), ('moddepth', res.moddepth)]:
        for k, v in d.items():
            s = dname+'/'+k
            summary_measures[s] = pop_summary(mse, v)
    summary_measures['mean/vs'] = pop_summary(mse, mean(res.raw_measures['vs'], axis=3))
    summary_measures['mean/onsettiness'] = pop_summary(mse, mean(res.raw_measures['onsettiness'], axis=3))
    for k, (low, high) in kwds.items():
        if k not in selected_axes and low!=high:
            summary_measures['param/'+k] = pop_summary(mse, reshape(res.raw.params[k], (M, M, num_params)))
    # Plot the data
    if smoothing:
        mse_summary = gaussian_filter(mse_summary, blur_width*M, mode='nearest')
        mse_summary = zoom(mse_summary, 100./M, order=1)
        mse_summary = gaussian_filter(mse_summary, blur_width*100., mode='nearest')
        mse_close = gaussian_filter(mse_close, blur_width*M, mode='nearest')
        mse_close = zoom(mse_close, 100./M, order=1)
        mse_close = gaussian_filter(mse_close, blur_width*100., mode='nearest')
        for k, v in summary_measures.items():
            v = gaussian_filter(v, blur_width*M, mode='nearest')
            v = zoom(v, 100./M, order=1)
            v = gaussian_filter(v, blur_width*100., mode='nearest')
            summary_measures[k] = v
    extent = (kwds[vx]+kwds[vy])
    num_rows = int(ceil(len(summary_measures)/4.0))+1
    curfig = figure(dpi=65, figsize=(14, (num_rows+1)*2.5))
    gs = GridSpec(num_rows, 4, height_ratios=[2]+[1]*(num_rows-1))

    def contourit():
        cs = contour(mse_summary, origin='lower',
                     levels=[15, 30, 45], colors='w',
                     extent=extent)
        clabel(cs, colors='w', inline=True, fmt='%d')
    
    subplot(gs[0, :2])
    imshow(mse_summary, origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, extent=extent)
    xlabel(sliders[vx].description)
    ylabel(sliders[vy].description)
    cb = colorbar()
    cb.set_label(error_func_name, rotation=270, labelpad=20)
    contourit()

    subplot(gs[0, 2:])
    imshow(100.*mse_close, origin='lower left', aspect='auto',
           interpolation='nearest', vmin=0, extent=extent)
    xlabel(sliders[vx].description)
    ylabel(sliders[vy].description)
    cb = colorbar()
    cb.set_label("Percent within cutoff", rotation=270, labelpad=20)
    contourit()
    
    for i, (k, v) in enumerate(summary_measures.items()):
        subplot(gs[1+i//4, i%4])
        vmin = 0
        vmax = 1
        if 'bmf' in k:
            vmax = 64
        if 'param' in k:
            vmin, vmax = kwds[k[6:]]
        imshow(v, origin='lower left', aspect='auto',
           interpolation='nearest', vmin=vmin, vmax=vmax, extent=extent)
        xlabel(sliders[vx].description)
        ylabel(sliders[vy].description)
        cb = colorbar()
        contourit()
        title(k)

    tight_layout()
# -

# ## GUI

# + {"hide_input": false}
sliders = OrderedDict([
    (spec['name'],
     ipw.FloatSlider(description=spec['description'], min=spec['min'], max=spec['max'],
                     step=spec['step'], value=spec['value'])) for spec in parameter_specs])
range_sliders = OrderedDict([
    (spec['name'],
     ipw.FloatRangeSlider(description=spec['description'], min=spec['min'], max=spec['max'],
                     step=spec['step'], value=(spec['min'], spec['max']))) for spec in parameter_specs])

detail_slider = ipw.Dropdown(description="Detail",
                             options=["Low", "Medium", "High"],
                             value='Low')

error_func_dropdown = ipw.Dropdown(description="Error function", options=error_functions.keys())

weighted_widget = ipw.Checkbox(description="Use weighted mean phase instead of peak", value=False)

def full_width_widget(widget):
    widget.layout.width = '95%'
    widget.style = {'description_width': '30%'}
    return widget

for slider in sliders.values()+range_sliders.values()+[detail_slider,
                                                       error_func_dropdown,
                                                       weighted_widget,
                                                       ]:
    full_width_widget(slider)

def savecurfig(fname):
    curfig.savefig(fname)
widget_savefig = save_fig_widget(savecurfig)

#########################################################################
# Model 1: MSE/MTF 2d maps
vars_mse_mtf = OrderedDict((k, v.description) for k, v in sliders.items())
vs2d_mse_mtf = VariableSelector(vars_mse_mtf, ['Horizontal axis', 'Vertical axis'], title=None,
                                initial={'Horizontal axis': 'alpha',
                                         'Vertical axis': 'beta'})
options2d_mse_mtf = {'var': vs2d_mse_mtf.widgets_vertical}

current_map2d_widgets = {}

def map2d(runmodel, vs2d):
    def f():
        params = vs2d.merge_selected(range_sliders, sliders)
        current_map2d_widgets.clear()
        current_map2d_widgets.update(params)
        params['detail'] = detail_slider
        params['interpolate_bmf'] = full_width_widget(ipw.Checkbox(description="Interpolate BMF",
                                                                   value=True))
        params['weighted'] = weighted_widget
        params['error_func'] = error_func_dropdown
        def plotter(**kwds):
            vx = vs2d.selection['Horizontal axis']
            vy = vs2d.selection['Vertical axis']
            return plot_map2d_mse_mtf((vx, vy), **kwds)
        i = ipw.interactive(plotter, dict(manual=True, manual_name="Run simulation"), **params)
        return no_continuous_update(i)
    return f

#########################################################################
# Model 2: population space    
    
def population_space():
    params = range_sliders.copy()
    params['weighted'] = weighted_widget
    params['detail'] = detail_slider
    params['error_func'] = error_func_dropdown
    params['interpolate_bmf'] = full_width_widget(ipw.Checkbox(description="Interpolate BMF",
                                                               value=False))
    # setup GUI
    i = grouped_interactive(plot_population_space, {'': params}, manual_name="Run simulation")
    return i

#########################################################################
# Model 3: Combined population / 2D map
vars_pop_map = OrderedDict((k, v.description) for k, v in sliders.items())
vs2d_pop_map = VariableSelector(vars_pop_map, ['Horizontal axis', 'Vertical axis'], title=None,
                                initial={'Horizontal axis': 'alpha',
                                         'Vertical axis': 'beta'})
options2d_pop_map = {'var': vs2d_pop_map.widgets_vertical}

current_pop_map_widgets = {}

def population_map():
    params = range_sliders.copy()
    current_pop_map_widgets.clear()
    current_pop_map_widgets.update(params)
    params['pop_summary'] = full_width_widget(
        ipw.Dropdown(description="Population summary method",
                     options=population_summary_methods.keys(),
                     value="Best"))
    params['detail'] = detail_slider
    params['weighted'] = weighted_widget
    params['smoothing'] = full_width_widget(
        ipw.Checkbox(description="Image smoothing", value=True))
    params['error_func'] = error_func_dropdown
    params['error_cutoff_deg'] = full_width_widget(
        ipw.FloatSlider(description="Error cutoff (deg)",
                        min=0, max=180, value=30, step=5))
    params['interpolate_bmf'] = full_width_widget(ipw.Checkbox(description="Interpolate BMF",
                                                               value=False))    
    def plotter(**kwds):
        vx = vs2d_pop_map.selection['Horizontal axis']
        vy = vs2d_pop_map.selection['Vertical axis']
        return plot_population_map((vx, vy), **kwds)
    i = ipw.interactive(plotter, dict(manual=True, manual_name="Run simulation"), **params)
    return no_continuous_update(i)

#########################################################################
# Construct and show GUI

models = [('2d map', map2d(simple_model, vs2d_mse_mtf), options2d_mse_mtf,
               [load_save_parameters_widget(current_map2d_widgets, 'saved_params_simple_map2d'),
                widget_savefig, progress_slider]),
          ('Population', population_space, {},
               [load_save_parameters_widget(range_sliders, 'saved_params_simple_population'),
                widget_savefig, progress_slider]),
          ('Population/map', population_map, options2d_pop_map,
               [load_save_parameters_widget(current_pop_map_widgets, 'saved_params_simple_popmap'),
                widget_savefig, progress_slider]),
         ]

# Create model explorer, and jump immediately to results page
modex = model_explorer(models)
# modex.widget_model_type.value = '2d map'
modex.widget_model_type.value = 'Population'
#modex.widget_model_type.value = 'Population/map'
modex.tabs.selected_index = 1
display(modex)
