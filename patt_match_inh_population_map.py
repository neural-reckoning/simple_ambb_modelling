# ---
# jupyter:
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
#     version: 2.7.12
# ---

# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")
from brian2 import *
numpy.set_printoptions(threshold=numpy.nan)
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from multiprocessing import *
from scipy.stats import norm,circstd,circmean
from scipy.signal import correlate2d
import time
import cProfile
from scipy.signal import fftconvolve

# +
__all__ = ['generate_random_mcalpine_et_al_2001_bds']

def fixup(s):
    s = s.replace(',', '.')
    s = s.split('\n')
    s = [map(float, w.split('    ')) for w in s if w]
    f, bitd = zip(*s)
    f = array(f)*kHz
    bitd = array(bitd)*usecond
    return f, bitd

fig_2a_means = '''
0,09455    707,10712
0,16542    520,84442
0,23318    361,37778
0,29635    277,76535
0,35333    232,09654
0,41458    182,66420
0,46000    163,59335
0,51884    205,06943
0,57556    148,14299
0,61844    113,97392
0,68096    147,91190
0,75553    117,48437
0,80553    121,18188
0,99987    109,52809
'''

fig_2a_means_plus_stds = '''
0,09879    1125,42432
0,19757    819,93372
0,30073    604,84766
0,39557    412,23495
0,49462    412,60233
0,59540    333,41052
0,68949    242,79839
0,78939    307,37531
0,89622    250,80063
0,97863    201,73302
1,09955    209,49567
1,23526    228,61478
1,34885    179,54718
1,75320    191,33490
'''

_, mean_bitd = fixup(fig_2a_means)
f, bitd_mean_plus_std = fixup(fig_2a_means_plus_stds)
std_bitd = bitd_mean_plus_std-mean_bitd

def generate_random_mcalpine_et_al_2001_bds(cf,N,std_factor=1.0):
    fmid = 0.5*(f[1:]+f[:-1])
    I = digitize(cf, fmid)
    mu = mean_bitd[I]*2*cf[0]*180.0
    sigma = std_bitd[I]*std_factor*2*cf[0]*180.0
    x_axis = np.arange(-180, 180, 360.0/N)
    ##Creating the 2-sided BIPD distribution
    dist_bipd=exp(-(mu-x_axis)**2/(2*sigma**2))+ exp(-(-mu-x_axis)**2/(2*sigma**2))
    dist_bipd=dist_bipd/max(dist_bipd)
    dist_bipd_recentered=np.concatenate((dist_bipd[int(N/2):int(N)],dist_bipd[0:int(N/2)]))
    #plot(x_axis,dist_bipd_recentered)
    #show()
    return dist_bipd_recentered

# -

def canonical_dens_plots(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fm,fc,m,ipd,tc,k,sigma,poisson_noise,dyn=False):
    fm=fm*Hz
    fc=fc*Hz
    tau=tau*ms
    fm_max=64*Hz
    n=int(1+np.ceil(3*tau*fm_max))

    if n%2==0:
        n+=1
    #spread=0.5
    #stre_inh=1
   
    #sig_con=2
    t = linspace(0, 1, n*Nx)*(1.0*n/fm)
    bipd = linspace(0, 2*n*pi, n*Ny, endpoint=False)%(2*pi)
    T, BIPD = meshgrid(t, bipd) 
    A = 0.5*(1-m*cos(2*pi*fm*T)) ##m   
    
    if dyn:
        PHI=(2*pi*fm*T+dphi_phase)%(2*pi)
    else:
        PHI =(ipd+dphi_phase)%(2*pi)
        
    if tc=='Cosine with exponent':
        TC=(cos((BIPD-PHI)/2))**k
    else:
        TC = (1.0/3)*(exp(-(BIPD-PHI)**2/(2*sigma**2))+
                   exp(-(BIPD-PHI+2*pi)**2/(2*sigma**2))+
                   exp(-(BIPD-PHI-2*pi)**2/(2*sigma**2)))
    
    layer1 = TC*A
    
    bipd_inf=int((Ny*n)*((n-1)/(2.0*n)))
    bipd_sup=int((Ny*n)*((n+1)/(2.0*n)))
    t_inf=int(Nx*n*(1-1.0/n))
    t_sup=int(Nx*n)
    
    if mcAlp_bipd:
        h_bipd=np.tile(generate_random_mcalpine_et_al_2001_bds([fc],Ny, std_factor=1.0),n)
        layer_weighted=layer1*h_bipd[...,np.newaxis]
    else:
        layer_weighted=layer1
     

    bipd_window = int(spread*(Ny))
    pix_t = (1.0/(Nx*fm))
    pix_bipd = 2*pi/(Ny)          
    ksize_t = int(3.*tau/pix_t)
        
    dt= arange(2*ksize_t+1)[::1]*pix_t 
    dbipd = arange(-bipd_window/2, bipd_window/2 +1)*pix_bipd
    DT, DBIPD = meshgrid(dt, dbipd)
    kernel0 = where(DT>ksize_t*pix_t, 0, exp(DT/tau))
    kernel0[:, kernel0.shape[1]/2] = 0
    kernel0[kernel0.shape[0]/2,:] = 0
    #layer_weighted=np.zeros_like(layer1)
    #layer_weighted[Ny+1,-layer1.shape[1]/4]=1

    if kernel_func=='Centrally weighted exp':
        for col in xrange(int(kernel0.shape[1]/2+1)):
            col_val=kernel0[0,col]
            for row in xrange(kernel0.shape[0]):
                kernel0[row,col]=col_val*exp(-0.001*abs(row-kernel0.shape[0]/2)**2/(2*sig_con**2))

    elif kernel_func=='Border weighted exp':
        for col in xrange(int(kernel0.shape[1]/2+1)):
            col_val=kernel0[0,col]
            for row in xrange(kernel0.shape[0]/2):
                kernel0[row+kernel0.shape[0]/2-2,col]=col_val*exp(-0.001*abs((kernel0.shape[0]-row)%(kernel0.shape[0]/2))**2/(2*sig_con**2))
                kernel0[row,col]=col_val*exp(-0.001*abs(row%(kernel0.shape[0]/2))**2/(2*sig_con**2))

    elif kernel_func=='Identity':
        kernel0=np.zeros(( dbipd.shape[0], dbipd.shape[0]))
        kernel0[int(dbipd.shape[0]/2),int(dbipd.shape[0]/2)]=1
    
    kernel0[dbipd.shape[0]/2,:]=0

    kernel=kernel0[::-1,::-1]
    kernel /= sum(kernel)
    inh_layer = fftconvolve(layer_weighted,kernel, mode='same')

    layer2= clip(layer_weighted-stre_inh*inh_layer,0, inf)        
    layer1=np.concatenate((layer1[bipd_inf:bipd_sup,t_inf:t_sup],layer1[bipd_inf:bipd_sup,t_inf:t_sup]),1)
    layer_weighted=np.concatenate((layer_weighted[bipd_inf:bipd_sup,t_inf:t_sup],layer_weighted[bipd_inf:bipd_sup,t_inf:t_sup]),1)
    layer_weighted_2=np.concatenate((layer2[bipd_inf:bipd_sup,t_inf:t_sup],layer2[bipd_inf:bipd_sup,t_inf:t_sup]),1)
    inh_layer=np.concatenate((inh_layer[bipd_inf:bipd_sup,t_inf:t_sup],inh_layer[bipd_inf:bipd_sup,t_inf:t_sup]),1)
    
    if poisson_noise:
        layer_weighted_2/=np.amax(layer_weighted_2)
        layer_weighted_2=np.random.poisson(layer_weighted_2*100.0)
        
   
    return layer1,layer_weighted_2,layer_weighted,kernel,inh_layer

def static_ipd_dens_plots_generation(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fm,fc,m,ipd,tc,k,sigma,poisson_noise,procnum=0,return_dict=None):
    layer_weighted=canonical_dens_plots(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fm,fc,m,ipd,tc,k,sigma,poisson_noise,dyn=False)[1]
    return_dict[procnum] =layer_weighted
    #return layer_weighted

def dyn_ipd_dens_plot_generation(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fm,fc,m,tc,sigma,k,poisson_noise):
    layer1,layer_weighted_2,layer_weighted,kernel,inh_layer=canonical_dens_plots(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fm,fc,m,0,tc,k,sigma,poisson_noise,dyn=True)
    return layer1,layer_weighted_2,layer_weighted,kernel,inh_layer

def training(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fc,num_training,m,tc,sigma,k,poisson_noise):
    ##Use of multiprocessing to make the generation of the training data faster
    n=2
    ipds=np.linspace(0,2*pi,num_training,endpoint=False)##check it out    
    num_fm=5
    training_patts=np.zeros((num_fm,num_training,Ny,n*Nx))
    proc=[]
    manager = Manager()
    return_dict = manager.dict()
    
    #for i in xrange(num_fm):
    #    for j in xrange(num_training):
    #       training_patts[i,j,:,:]=static_ipd_dens_plots_generation(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,2**(i+2),fc,ipds[j],tc,k,sigma,poisson_noise)#,i*num_training+j,return_dict,)))
    
    
    
    ##fix multiprocessin geither pool or None
    for i in xrange(num_fm):
        for j in xrange(num_training):
            proc.append(Process(target=static_ipd_dens_plots_generation,args=(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,2**(i+2),fc,m,ipds[j],tc,k,sigma,poisson_noise,i*num_training+j,return_dict,)))
        
    for i in xrange(num_fm):
        for j in xrange(num_training):
            proc[i*num_training+j].start()
        
    for i in xrange(num_fm):
        for j in xrange(num_training):
            proc[i*num_training+j].join()
        
    for i in xrange(num_fm):
        for j in xrange(num_training):
            training_patts[i,j,:,:]= return_dict[i*num_training+j]
         
    
    return training_patts

def testing_dyn_ipd(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fc,m,tc,sigma,k,poisson_noise):   
    #Generating training data for all modulation frequency
    n=2
    num_fm=5
    testing_patts=np.zeros((num_fm,Ny,2*Nx))

    for i in xrange(num_fm):
        testing_patts[i,:]=dyn_ipd_dens_plot_generation(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,2**(i+2),fc,m,tc,sigma,k,poisson_noise)[1]
        
            
    return testing_patts

def testing(dphi,Ny,num_training,test_img,training_patts,similarity_method,decision_window,width_frame):
    dphi_phase=dphi*360.0/Ny
    num_fm=5
    scores=np.zeros((num_fm,num_training))
    std_scores=np.zeros((num_fm))
    results=np.zeros(num_fm)
    #hemi_ratio_tests=np.zeros(num_fm)
    #hemi_ratio_trains=np.zeros((num_fm,num_training))
    #results_hemi=np.zeros(num_fm)
    sum_tests=np.zeros((num_fm,Ny))
    sum_train_dec_winds=np.zeros((num_fm,num_training,Ny))
    abs_diff_ratio=np.zeros((num_fm,num_training))
    test_mat=np.zeros((num_fm,Ny,width_frame))
    train_mat=np.zeros((num_fm,num_training,Ny,width_frame))
    
    
    ##Create mini function to refactor that
 
    if similarity_method=='Euclidean distance':
        
        ##see if you can vectorize that
        for i in xrange(num_fm):
            #btest=test_img[i][:,decision_window[0]:decision_window[1]]>0
            for j in xrange(num_training):
                #btrain=training_patts[i,j,:,decision_window[0]:decision_window[1]]>0
                if np.any(training_patts[i,j,:,decision_window[0]:decision_window[1]]) and np.any(test_img[i][:,decision_window[0]:decision_window[1]]): 
                    scores[i,j]=np.linalg.norm((test_img[i][:,decision_window[0]:decision_window[1]]/
                                               np.linalg.norm(test_img[i][:,decision_window[0]:decision_window[1]]))
                                               -(training_patts[i,j,:,decision_window[0]:decision_window[1]]/
                                                 np.linalg.norm(training_patts[i,j,:,decision_window[0]:decision_window[1]])))
        for i in xrange(num_fm):
            scores[i]/=np.max(scores[i])
        
        scores=1-scores
    
    elif similarity_method=='2D dot product':
        for i in xrange(num_fm):
            #btest=test_img[i][:,decision_window[0]:decision_window[1]]
            for j in xrange(num_training):              
                #btrain=training_patts[i,j,:,decision_window[0]:decision_window[1]]>0
                if np.any(test_img[i][:,decision_window[0]:decision_window[1]]) and np.any(training_patts[i,j,:,decision_window[0]:decision_window[1]]):              
                    scores[i,j]=sum(test_img[i][:,decision_window[0]:decision_window[1]]
                                     *training_patts[i,j,:,decision_window[0]:decision_window[1]])
                                      
                    scores[i,j]/=(np.linalg.norm(test_img[i][:,decision_window[0]:decision_window[1]])*
                                  np.linalg.norm(training_patts[i,j,:,decision_window[0]:decision_window[1]])) 
                                
    elif similarity_method==' ':
        
        for i in xrange(num_fm):
            for j in xrange(num_training):
                sum_test=sum(test_img[i][:,decision_window[0]:decision_window[1]],1)
                sum_train=sum(training_patts[i,j,:,decision_window[0]:decision_window[1]],1)
                scores[i,j]=np.dot(sum_test,sum_train)
                #btest=sum_test>0
                #btrain=sum_train>0
                if np.any(sum_test) and np.any(sum_train):
                    scores[i,j]/=np.linalg.norm(sum_test)*np.linalg.norm(sum_train)
       
    else:
        print 'error'
    results=np.argmax(scores,1)*360.0/num_training + dphi_phase
    results%=360
    sum_sin=np.zeros((num_fm))
    sum_cos=np.zeros((num_fm))
    ipds=np.linspace(0,2*pi,num_training,endpoint=False)
    for k in xrange(num_fm):
        for j in xrange(num_training):
            sum_sin[k]+=np.sin(ipds[j]+0*dphi_phase*pi/180.0)*scores[k,j]
            sum_cos[k]+=np.cos(ipds[j]+0*dphi_phase*pi/180.0)*scores[k,j]
        std_scores[k]=sqrt(-np.log((1.0/sum(scores[k,:])**2)*(sum_sin[k]**2+sum_cos[k]**2)))*180.0/pi
     
                            
    return scores,results,std_scores

def simple_browsing(tau,spread,stre_inh,k):
    exp_data=[37, 40, 62, 83, 115]
    Nx=250
    Ny=100
    dphi=0
    num_training=20
    mcAlp_bipd=True
    sig_con=2
    fc=500
    sigma=1
    tc='Cosine with exponent'
    poisson_noise=False
    kernel_func='Constant'
    similarity_method='2D dot product'
    m=1
    n=2   
    dphi_phase=0
    dphi_pix=0
    training_patts=training(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fc,num_training,m,tc,sigma,k,poisson_noise)
    testing_patts=testing_dyn_ipd(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fc,m,tc,sigma,k,poisson_noise)

    decision_window_pix=[Nx,2*Nx-1] 
    
    res=testing(dphi_pix,Ny,num_training,testing_patts,training_patts,similarity_method,decision_window_pix,decision_window_pix[1]-decision_window_pix[0])[1]
    mse_res=np.mean((res-exp_data)**2)
    if not np.any(res):
        mse_res=nan
    return mse_res

def pop_maps(arr_tau,arr_spread,arr_stre_inh,arr_k):
    
    res=np.zeros(( size(arr_tau),size(arr_spread),size(arr_stre_inh),size(arr_k)))
    w=widgets.FloatProgress(
    value=0,
    min=0,
    max=100.0,
    step=0.1,
    description='Loading:',
    bar_style='info',
    orientation='horizontal',
    layout=widgets.Layout(width='100%')
    )
    display(w)
    cmpt_max=size(res)
    print cmpt_max
    cmpt=0
    for i in xrange(size(arr_tau)):
        for j in xrange(size(arr_spread)):
            for k in xrange(size(arr_stre_inh)):
                for l in xrange(size(arr_k)):
                    res[i,j,k,l]=simple_browsing(arr_tau[i],arr_spread[j],arr_stre_inh[k],arr_k[l])
                    cmpt+=1.0
                    w.value=(cmpt/cmpt_max)*100
   
    print res
    
    save('res.npy',res)            

# +
def comparison_decoder(tau,spread,stre_inh,k):

    arr_tau=np.arange(tau[0],tau[1],2)
    print arr_tau
    arr_spread=np.arange(spread[0],spread[1],0.2)
    print arr_spread
    arr_stre_inh=np.arange(stre_inh[0],stre_inh[1],1)
    print arr_stre_inh
    arr_k=np.arange(k[0],k[1],2)
    print arr_k
    
    pop_maps(arr_tau,arr_spread,arr_stre_inh,arr_k)
    

w_in=widgets.interactive(comparison_decoder,tau=widgets.FloatRangeSlider(value=[1,10],min=1, max=10, step=1,description='Time constant exp decay kernel (width kernel)')
,spread=widgets.FloatRangeSlider(value=[0.2,1],min=0.2,max=1,step=0.1,description='height of the kernel')
,stre_inh=widgets.FloatRangeSlider(value=[0,4],min=0,max=5,step=0.5,description='Strength of inhibition')
,k=widgets.IntRangeSlider(value=[0,10],min=0, max=10, step=2,description='Exponent cosine k'))


for child in w_in.children:
    if isinstance(child, widgets.ValueWidget):
        child.layout.width = '100%'
        child.style = {'description_width': '30%'}
        child.continuous_update = False

display(w_in)

# +
def visualization(filename,i_tau,i_spread,i_stre_inh,i_k):
    res=load(filename)
    ind_tau=int(res.shape[0]*i_tau)
    ind_spread=int(res.shape[1]*i_spread)
    ind_stre_inh=int(res.shape[2]*i_stre_inh)
    ind_k=int(res.shape[3]*i_k)
    
    figure(figsize=(20,20))
    #gs = Gridspec(3,2)
    subplot(321)
    imshow(res[:,:,ind_stre_inh,ind_k],origin='lower left')
    colorbar()
    
    subplot(322)
    imshow(res[:,ind_spread,:,ind_k],origin='lower left')
    colorbar()
    
    subplot(323)
    imshow(res[:,ind_spread,ind_stre_inh,:],origin='lower left')
    colorbar()
    
    subplot(324)
    imshow(res[ind_tau,:,:,ind_k],origin='lower left')
    colorbar()
    
    subplot(325)
    imshow(res[ind_tau,:,ind_stre_inh,:],origin='lower left')
    colorbar()
    
    subplot(326)
    imshow(res[ind_tau,ind_spread,:,:],origin='lower left')
    colorbar()
    
w=widgets.interactive(visualization,filename='res.npy',i_tau=widgets.FloatSlider(value=0,min=0, max=1, step=0.1,description='index tau')
,i_spread=widgets.FloatSlider(value=0,min=0,max=1,step=0.1,description='index spread')
,i_stre_inh=widgets.FloatSlider(value=0,min=0,max=1,step=0.1,description='index strength of inhibition')
,i_k=widgets.FloatSlider(value=0,min=0, max=1, step=0.1,description='index exponent cosine k'))


for child in w.children:
    if isinstance(child, widgets.ValueWidget):
        child.layout.width = '100%'
        child.style = {'description_width': '30%'}
        child.continuous_update = False

display(w)
    
    
    
    
    
# -


