'''
The core single neuron model.
'''
from brian2 import *
from scipy.interpolate import interp1d
import joblib

__all__ = ['dietz_fm', 'dietz_phase', 'dietz_phase_std', 'simple_model', 'simple_model_results']

dietz_fm = array([4, 8, 16, 32, 64])*Hz
dietz_phase = array([37, 40, 62, 83, 115])*pi/180
dietz_phase_std = array([46, 29, 29, 31, 37])*pi/180

# Known warnings that we expect, so suppress them
BrianLogger.suppress_name('resolution_conflict')
BrianLogger.suppress_name('invalid_values')

def normed(X, *args):
    m = max(amax(abs(Y)) for Y in (X,)+args)
    return X/m

mem = joblib.Memory(location='.', bytes_limit=10*1024**3, verbose=0) # 10 GB max cache


class Results(object): # simple heap class, we'll add attributes to it
    def __init__(self, **kwds):
        for k, v in kwds.items():
            setattr(self, k, v)


@mem.cache(ignore=['update_progress'])
def simple_model(N, params, record=None, update_progress=None,
                 fm=None, minimum_initial_time=100*ms):
    seed(3402348923) # for reproducibility
    if fm is None:
        fm = dietz_fm
    orig_fm = fm
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
    # Accumulation variables
    start_time : second
    end_time : second
    do_accumulate = 1.0*int(t>=start_time and t<end_time) : 1
    accum_sum_out : 1
    accum_sum_out_rising : 1
    accum_sum_out_falling : 1
    accum_argmax_out : second
    accum_max_out : 1
    accum_weighted_sum_cos_phase : 1
    accum_weighted_sum_sin_phase : 1
    '''
    G = NeuronGroup(N*len(fm), eqs, method='euler', dt=0.1*ms)
    rr = G.run_regularly('''
        accum_sum_out += out*do_accumulate
        phase = (2*pi*fm*t)%(2*pi)
        accum_sum_out_rising += out*int(phase<pi)*do_accumulate
        accum_sum_out_falling += out*int(phase>=pi)*do_accumulate
        accum_weighted_sum_cos_phase += out*cos(phase)*do_accumulate
        accum_weighted_sum_sin_phase += out*sin(phase)*do_accumulate
        is_larger = out>accum_max_out and do_accumulate>0
        accum_max_out = int(not is_larger)*accum_max_out+int(is_larger)*out
        accum_argmax_out = int(not is_larger)*accum_argmax_out+int(is_larger)*t
        ''',
        when='end')
    params = params.copy()
    for k, v in params.items():
        if isinstance(v, tuple) and len(v)==2:
            low, high = v
            params[k] = v = rand(N)*(high-low)+low
    params2d = params.copy()
    for k, v in params2d.items():
        if isinstance(v, ndarray) and v.size>1:
            v = reshape(v, v.size)
            fm, v = meshgrid(orig_fm, v) # fm and v have shape (N, len(dietz_fm))
            fm.shape = fm.size
            v.shape = v.size
            params2d[k] = v
    params2d['fm'] = fm
    G.set_states(params2d)
    G.tauihc_ms['tauihc_ms<min_tauihc/ms'] = 0
    G.Q = 1
    net = Network(G, rr)
    # Calculate how long to run the simulation
    period = 1/fm
    num_initial_cycles = ceil(minimum_initial_time/period) # at least one period and at least that time
    start_time = num_initial_cycles*period
    end_time = (num_initial_cycles+1)*period
    duration = amax(end_time)
    G.start_time = start_time
    G.end_time = end_time
    # Run the simulation
    if isinstance(update_progress, basestring):
        report_period = 10*second
    else:
        report_period = 1*second
    if record:
        M = StateMonitor(G, record, record=True)
        net.add(M)
    net.run(duration, report=update_progress, report_period=report_period)
    G.accum_sum_out['accum_sum_out<1e-10'] = 1
    for name in ['accum_sum_out_rising', 'accum_sum_out_falling',
                 'accum_argmax_out', 'accum_max_out', 'accum_weighted_sum_cos_phase',
                 'accum_weighted_sum_sin_phase']:
        if name=='accum_argmax_out':
            u = second
        else:
            u = 1
        getattr(G, name)['accum_sum_out>1e10'] = 0*u
    G.accum_sum_out['accum_sum_out>1e10'] = 1
    c = G.accum_weighted_sum_cos_phase[:]
    s = G.accum_weighted_sum_sin_phase[:]
    weighted_phase = (angle(c+1j*s)+2*pi)%(2*pi)
    vs = sqrt(c**2+s**2)/G.accum_sum_out[:]
    mean_fr = G.accum_sum_out[:]/((end_time-start_time)/G.dt)
    onsettiness = 0.5*(1+(G.accum_sum_out_rising[:]-G.accum_sum_out_falling[:])/G.accum_sum_out[:])
    res = Results(
        accum_argmax_out=G.accum_argmax_out[:],
        accum_max_out=G.accum_max_out[:],
        weighted_phase=weighted_phase,
        vs=vs,
        mean_fr=mean_fr,
        onsettiness=onsettiness,
        params=params,
        start_time=start_time,
        end_time=end_time,
        )
    if record:
        for name in record:
            v = getattr(M, name)[:, :]
            v.shape = (N, len(dietz_fm), -1)
            setattr(res, name, v)
        res.t = M.t[:]
    return res

@mem.cache
def simple_model_results(N, out, error_func, weighted=False, interpolate_bmf=False,
                         shape=None, fm=None, target_phase=None):
    if fm is None:
        fm = dietz_fm
        dietz_fm_idx = arange(len(dietz_fm))
    else:
        dietz_fm_idx = argmin((fm[:, newaxis]-dietz_fm[newaxis, :])**2, axis=0)
    n_fm = len(fm)
    if shape is None:
        shape = (N,)
    if target_phase is None:
        target_phase = dietz_phase
    out_peak = out.accum_argmax_out
    peak_fr = out.accum_max_out
    weighted_phase = out.weighted_phase
    vs = out.vs
    mean_fr = out.mean_fr
    onsettiness = out.onsettiness
    out_peak.shape = peak_fr.shape = weighted_phase.shape = vs.shape = mean_fr.shape = onsettiness.shape = (N, n_fm)
    if weighted:
        peak_phase = weighted_phase
    else:
        peak_phase = (out_peak*2*pi*fm[newaxis, :]) % (2*pi) # shape (N, n_fm)
    max_peak_fr = amax(peak_fr, axis=1)[:, newaxis]
    max_peak_fr[max_peak_fr==0] = 1
    max_mean_fr = amax(mean_fr, axis=1)[:, newaxis]
    max_mean_fr[max_mean_fr==0] = 1
    norm_peak_fr = peak_fr/max_peak_fr
    norm_mean_fr = mean_fr/max_mean_fr
    mse = error_func(target_phase[newaxis, :], peak_phase[:, dietz_fm_idx]) # sum over fm, mse has shape N
    mse_norm = (mse-amin(mse))/(amax(mse)-amin(mse))
    peak_bmf = asarray(fm)[argmax(norm_peak_fr, axis=1)]
    mean_bmf = asarray(fm)[argmax(norm_mean_fr, axis=1)]
    vs_bmf = asarray(fm)[argmax(vs, axis=1)]
    onsettiness_bmf = asarray(fm)[argmax(onsettiness, axis=1)]
    peak_moddepth = 1-amin(norm_peak_fr, axis=1)
    mean_moddepth = 1-amin(norm_mean_fr, axis=1)
    vs_moddepth = amax(vs, axis=1)-amin(vs, axis=1)
    onsettiness_moddepth = amax(onsettiness, axis=1)-amin(onsettiness, axis=1)
    # interpolated bmf
    if interpolate_bmf:
        fm_interp = linspace(4, 64, 100)
        for cx in xrange(N):
            for bmf, fr in [(peak_bmf, norm_peak_fr),
                            (mean_bmf, norm_mean_fr),
                            (vs_bmf, vs),
                            (onsettiness_bmf, onsettiness)]:
                cur_fr = fr[cx, :]
                fr_interp_func = interp1d(fm, cur_fr, kind='quadratic')
                bmf[cx] = fm_interp[argmax(fr_interp_func(fm_interp))]
    raw_measures = {'peak': peak_fr, 'mean': mean_fr, 'vs': vs, 'onsettiness': onsettiness}
    norm_measures = {'peak': norm_peak_fr, 'mean': norm_mean_fr, 'vs': vs, 'onsettiness': onsettiness}
    bmf = {'peak': peak_bmf, 'mean': mean_bmf, 'vs': vs_bmf, 'onsettiness': onsettiness_bmf}
    moddepth = {'peak': peak_moddepth, 'mean': mean_moddepth, 'vs': vs_moddepth, 'onsettiness': onsettiness_moddepth}
    for img in [peak_phase]+raw_measures.values()+norm_measures.values():
        img.shape = shape+(n_fm,)
    for img in [mse, mse_norm]+bmf.values()+moddepth.values():
        img.shape = shape
    res = Results(
        peak_phase=peak_phase,
        mse=mse,
        mse_norm=mse_norm,
        raw_measures=raw_measures,
        norm_measures=norm_measures,
        bmf=bmf,
        moddepth=moddepth,
        raw=out,
        )
    return res
