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
#     version: 2.7.12
# ---

# + {"hide_input": false}
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

def hemispheric_decoder(activity_vec):
    
    #Output activity vector
    half_dim=int(0.5*activity_vec.shape[0])

    #Split it in 2-halves :ipsi and contra
    act_ipsi=activity_vec[0:half_dim]
    
    act_contra=activity_vec[half_dim:activity_vec.shape[0]]
    
    #sum each half
    sum_ipsi=sum(act_ipsi)

    sum_contra=sum(act_contra)
    
    #And compute the ratio
    ratio=(sum_ipsi-sum_contra)/sum(activity_vec)

    return ratio
    

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
 
    
    #std_scores=std(scores,1)
    '''
    ### Hemispheric decoder
    for i in xrange(num_fm):
            sum_tests[i]=sum(test_mat[i],1)
            hemi_ratio_tests[i]=hemispheric_decoder(sum_tests[i])
            
    for i in xrange(num_fm):
        for j in xrange(num_training):
            sum_train_dec_winds[i,j]=sum(train_mat[i,j],1)
            hemi_ratio_trains[i,j]=hemispheric_decoder(sum_train_dec_winds[i,j,:])
    
    abs_diff_ratio = abs(hemi_ratio_tests[:, newaxis]-hemi_ratio_trains) ##that too
    results_hemi=argmin(abs_diff_ratio,1)*360.0/(num_training-1) + dphi_phase
    '''
    
    #if procnum!=None:
    #    return_dict[procnum]=scores
        
    
                            
    return scores,results,std_scores

def testing_sliced(dphi,Nx,Ny,num_training,test_img,training_patts,decision_window):
    dphi_phase=dphi*360.0/Ny
    num_fm=5
    width_slices=decision_window[1]-decision_window[0]
    num_slices=Nx
    scores=np.zeros((num_fm,num_slices,num_training))
    results=np.zeros(num_fm)
    slices=np.zeros(num_fm)
    sum_tests=np.zeros((num_fm,Ny))
    abs_diff_ratio=np.zeros((num_fm,num_training))
    
    for i in xrange(num_fm):
        for j in xrange(num_training):
            for k in xrange(num_slices):
                if np.any(test_img[i][:,decision_window[0]:decision_window[1]]) and np.any(training_patts[i,j,:,Nx-width_slices+k:k+Nx]):              
                    scores[i,k,j]=sum(test_img[i][:,decision_window[0]:decision_window[1]]
                                     *training_patts[i,j,:,Nx-width_slices+k:k+Nx])

                    scores[i,k,j]/=(np.linalg.norm(test_img[i][:,decision_window[0]:decision_window[1]])
                                    *np.linalg.norm(training_patts[i,j,:,Nx-width_slices+k:k+Nx]))  

    for i in xrange(num_fm):
        slices[i],results[i]=np.unravel_index(scores[i].argmax(), scores[i].shape)
        results[i]*=360.0/(num_training)
        
    scores_train=np.amax(scores,1)

    return scores_train,results#,scores_train
    

def stats_process(img,num_fm,Nx,num_training,dphi_phase):
    
        max_simi=np.zeros((num_fm,Nx))
        arg_max_simi=np.zeros((num_fm,Nx))
        std_circ=np.zeros((num_fm,Nx))
        mean_circ=np.zeros((num_fm,Nx))
        arg_mm=np.zeros(num_fm)
        time_max=np.zeros(num_fm)
        sum_sin=np.zeros((num_fm,Nx))
        sum_cos=np.zeros((num_fm,Nx))
        ipds=np.linspace(0,2*pi,num_training,endpoint=False)
        
        for k in xrange(num_fm):
            for i in xrange(Nx):
                for j in xrange(num_training):
                    sum_sin[k,i]+=np.sin(ipds[j]+0*dphi_phase*pi/180.0)*img[k][j,i]
                    sum_cos[k,i]+=np.cos(ipds[j]+0*dphi_phase*pi/180.0)*img[k][j,i]
                mean_circ[k,i]=(arctan2(sum_sin[k,i],sum_cos[k,i])*180.0/pi)
                std_circ[k,i]=sqrt(-np.log((1.0/sum(img[k][:,i])**2)*(sum_sin[k,i]**2+sum_cos[k,i]**2)))*180.0/pi
                max_simi[k,i]=np.amax(img[k][:,i])*180.0
                arg_max_simi[k,i]=np.argmax(img[k][:,i])    
                if mean_circ[k,i]<0:
                    mean_circ[k,i]=360+mean_circ[k,i]

            time_max[k]=argmax(max_simi[k,:])

            arg_mm[k]=arg_max_simi[k,int(time_max[k])]
        return std_circ,mean_circ,max_simi,arg_mm

def manual_inhibition(tau,spread,kernel_func,mcAlp_bipd,Nx,Ny,fm,fc,tc,sigma):
    fm=fm*Hz
    fc=fc*Hz
    tau=tau*ms
    n=2
    t = linspace(0, 1, n*Nx)*(1.0*n/fm)
    bipd = linspace(0, 2*pi, Ny, endpoint=False)%(2*pi)
    T, BIPD = meshgrid(t, bipd) 
    A = 0.5*(1-cos(2*pi*fm*T)) ##m   
    
    PHI=(2*pi*fm*T)%(2*pi)
   
    if tc=='Cosine with exponent':
        TC=(cos((BIPD-PHI)/2))**4
    else:
        TC = (1.0/3)*(exp(-(BIPD-PHI)**2/(2*sigma**2))+
                   exp(-(BIPD-PHI+2*pi)**2/(2*sigma**2))+
                   exp(-(BIPD-PHI-2*pi)**2/(2*sigma**2)))
    
    layer1 = TC*A
    
    if mcAlp_bipd:
        h_bipd=np.tile(generate_random_mcalpine_et_al_2001_bds([fc],Ny, std_factor=1.0),1)
        layer_weighted=layer1*h_bipd[...,np.newaxis]
    else:
        layer_weighted=layer1
     

    bipd_window = int(spread*(Ny))
    pix_t = (1.0/(Nx*fm))
    pix_bipd = 2*pi/(Ny)          
    ksize_t = int(3.*tau/pix_t)
    dt= arange(2*ksize_t+1)[::1]*pix_t 
    dbipd = arange(-floor(bipd_window/2), floor(bipd_window/2) +1)*pix_bipd
    DT, DBIPD = meshgrid(dt, dbipd)
    kernel0 = where(DT>ksize_t*pix_t, 0, exp(DT/tau))
    kernel0[:, kernel0.shape[1]/2] = 0
    kernel=kernel0[::-1,::-1]
    kernel /= sum(kernel)
    
    inh_layer=np.zeros((Ny, Nx))

    for i in xrange(Nx):
        for j in xrange(Ny):
            s=0
            for k in xrange(kernel0.shape[1]):
                for l in xrange(kernel0.shape[0]):
                    if j-l>=0 and i-k>=0:                
                        s+=layer1[j-1-l,i-1-k]*exp(-abs(k*((2.0/(fm*second))/Nx)/(tau*Hz)))
            inh_layer[j,i]=clip(layer1[j,i]-0.015*s,0,inf)
    
        
   
    return inh_layer

def simple_browsing(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi,Nx,Ny,fc,num_training,m,tc,sigma,k,poisson_noise,similarity_method,decision_window,width_frame,index_mf):

    n=2
    start = time.time()
    #print "Start timer"
    
    dphi_pix=int(dphi*Ny)
    dphi_phase=dphi_pix*360.0/Ny
    
    #print 'Phase difference at t=0: '+str(dphi_phase)
    
    
    training_patts=training(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fc,num_training,m,tc,sigma,k,poisson_noise)
    testing_patts=testing_dyn_ipd(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fc,m,tc,sigma,k,poisson_noise)
    
    decision_window_pix_inf=int(decision_window[0]*n*Nx/(2*pi*n))
    deg_dec_wind_inf=(decision_window[0]%(2*pi))*(180.0)/pi + dphi_phase
    deg_dec_wind_inf%=360
    deg_dec_wind_sup=(decision_window[1]%(2*pi))*(180.0)/pi + dphi_phase
    deg_dec_wind_sup%=360
    print  deg_dec_wind_inf, deg_dec_wind_sup
    
    decision_window_pix_sup=int(decision_window[1]*n*Nx/(2*pi*n))
    decision_window_pix=[decision_window_pix_inf, decision_window_pix_sup] 
    print decision_window_pix
    
    
    res=testing(dphi_pix,Ny,num_training,testing_patts,training_patts,similarity_method,decision_window_pix,decision_window_pix[1]-decision_window_pix[0])
    
    num_fm=5
  
    ##Constructing graph 9
    width_frame_pix=int(width_frame*Nx)
    print 'number of pixels: '+ str(width_frame_pix)

    BIPD_vs_time=np.zeros((num_fm,num_training,Nx))
    BIPD_vs_time_sliced=np.zeros((num_fm,num_training,Nx))
    for i in xrange(Nx):
            BIPD_vs_time[:,:,i]=testing(dphi_pix,Ny,num_training,testing_patts,training_patts,similarity_method,
                                        [Nx-width_frame_pix+i,Nx+i],width_frame_pix)[0]
            #BIPD_vs_time_sliced[:,:,i]=testing_sliced(dphi_pix,Nx,Ny,num_training,testing_patts,training_patts,
             #                                         [Nx-width_frame_pix+i,Nx+i])[0]
    
    fm=2**(index_mf+2)
    dynIPD4=dyn_ipd_dens_plot_generation(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_pix,Nx,Ny,fm,fc,m,tc,sigma,k,poisson_noise)
    fm_array=[4,8,16,32,64]*Hz
    
    
    
    
    '''man=manual_inhibition(tau,spread,kernel_func,mcAlp_bipd,Nx,Ny,fm,fc,tc,sigma)
    figure(1)
    imshow(man,origin='lower left')
    '''
  
    figure(figsize=(30,20))
 
    
    gs = GridSpec(4, 4) 
    subplot(gs[0,0])
    title('Density plot, fm='+ str(fm)+' Hz before BIPD weighting')
    imshow(dynIPD4[0],origin='lower left', interpolation='nearest', aspect='auto')
    colorbar()

    subplot(gs[0,1])
    if mcAlp_bipd==True:
        dist_bipd= generate_random_mcalpine_et_al_2001_bds([fc*Hz],Ny,std_factor=1.0)
    else:
        dist_bipd=np.ones(Ny)
    
    x_axis= np.arange(0, 360, 360.0/Ny)
    title('Distribution of BIPD when fc='+str(fc)+' Hz')
    plot(x_axis,dist_bipd)
    xlabel('BIPD (deg)')
    ylabel('Normalized frequency')
    
    
    subplot(gs[0,2])
    title('Density plot, fm='+ str(fm)+' Hz after BIPD weighting')
    imshow(dynIPD4[2],origin='lower left', interpolation='nearest', aspect='auto',extent=[0,2*(1.0/fm),0,360])
    colorbar()
    xlabel('Time (second)')
    ylabel('BIPD (deg)')
    
    subplot(gs[0,3])
    title('Kernel, fm='+str(fm)+' Hz')
    imshow(dynIPD4[3][::-1,::-1],origin='lower left', interpolation='nearest', aspect='auto')
    colorbar()
    #xlabel('Time (second)')
    ylabel('BIPD (deg)')
    
    subplot(gs[1,0])
    title('Inhibition layer, fm='+ str(fm) +'Hz after BIPD weighting')
    imshow(dynIPD4[4],origin='lower left', interpolation='nearest', aspect='auto',extent=[0,2*(1.0/fm),0,360])
    colorbar()
    xlabel('Time (second)')
    ylabel('BIPD (deg)')
    
    subplot(gs[1,1])
    title('Density plot after Kernel inhibition, fm='+ str(fm) +'Hz after BIPD weighting')
    img=dynIPD4[1]
    

    #noise_mask = numpy.random.poisson(img*PEAK)
    imshow(dynIPD4[1],origin='lower left', interpolation='nearest', aspect='auto',extent=[0,2*(1.0/fm),0,360])
    colorbar()
    xlabel('Time (second)')
    ylabel('BIPD (deg)')
    
    subplot(gs[1,2])
    ntrain=int((num_training/4.0)*3)
    title('Example training data, ipd=' + str(ntrain*360.0/(num_training))+ 'deg, fm=' +str(fm)+' Hz')
    imshow(training_patts[index_mf,ntrain],origin='lower left', interpolation='nearest', aspect='auto',extent=[0,2*(1.0/fm),0,360])
    colorbar()
    xlabel('Time (second)')
    ylabel('BIPD')
    
    subplot(gs[1,3])
    title('Decision window, fm= '+str(fm)+' Hz')
    imshow(testing_patts[index_mf][:,decision_window_pix[0]:decision_window_pix[1]],origin='lower left', interpolation='nearest', aspect='auto')
    colorbar()
    xlabel('Time')
    ylabel('BIPD (deg)')
    
    subplot(gs[2,0])
    title('Scoring of the dyn IPD with the training pattern with BIPD on y axis')
    imshow(transpose(res[0]),origin='lower left', interpolation=None, aspect='auto',extent=[4,64,0,360])
    ylabel('BIPD (deg)')
    xlabel('Modulation frequency')
    colorbar()


   
    
    subplot(gs[2,1])
    title('Scoring dyn IPD with training patts for all dt,fm='+ str(fm)+' Hz')
    imshow(BIPD_vs_time[index_mf],origin='lower left', interpolation=None, aspect='auto')#,extent=[0,n/fm,0,360])
    ylabel('BIPD')
    colorbar()
    
    subplot(gs[2,2])
    title('Scoring dyn IPD with training patts for all dt whith sliced training data,fm='+ str(fm)+' Hz')
    imshow(BIPD_vs_time[index_mf],origin='lower left', interpolation=None, aspect='auto')#,extent=[0,n/fm,0,360])
    ylabel('BIPD')
    colorbar()

    
    stats=stats_process(BIPD_vs_time,num_fm,Nx,num_training,dphi_phase)
    stats_sliced=stats_process(BIPD_vs_time,num_fm,Nx,num_training,dphi_phase)    
    print res[2]
    print stats[3]*360.0/num_training  
    print stats_sliced[3]*360.0/num_training  
    c=np.zeros(Nx)
    for i in xrange(Nx):
        c[i]=0.5*(1-cos(2*pi*fm*i*((1.0/fm)/Nx)))
        
    subplot(gs[2,3])
    title('Standard deviation function of time, fm='+str(fm)+' Hz')
    plot(np.multiply(range(Nx),((1.0*1/fm)/Nx)*360*1),360*c,label='Envelope')
    plot(np.multiply(range(Nx),((1.0*1/fm)/Nx)*360*1),stats[0][index_mf],label='Standard deviation')
    plot(np.multiply(range(Nx),((1.0*1/fm)/Nx)*360*1),stats[1][index_mf],label='Mean')
    plot(np.multiply(range(Nx),((1.0*1/fm)/Nx)*360*1),stats[2][index_mf],label='Max Similarity')
    xlabel('Time')
    ylabel('BIPD')
    legend()
    
    subplot(gs[3,0])
    title('Standard deviation function of time,training patts sliced, fm='+str(fm)+' Hz')
    plot(np.multiply(range(Nx),((1.0*1/fm)/Nx)*360*1),360*c,label='Envelope')
    plot(np.multiply(range(Nx),((1.0*1/fm)/Nx)*360*1),stats_sliced[0][index_mf],label='Standard deviation')
    plot(np.multiply(range(Nx),((1.0*1/fm)/Nx)*360*1),stats_sliced[1][index_mf],label='Mean')
    plot(np.multiply(range(Nx),((1.0*1/fm)/Nx)*360*1),stats_sliced[2][index_mf],label='Max Similarity')
    xlabel('Time')
    ylabel('BIPD')
    legend()
    
    
    subplot(gs[3,1])
    title('Phase_fm curve')
    errorbar(fm_array, res[1],yerr=res[2], fmt='-o', label=str(similarity_method))
    plot(fm_array, stats[3]*360.0/(num_training), '-o', label='Argmax')
    #plot(fm_array, stats_sliced[3]*360.0/(num_training), '-o', label='Argmax sliced')
    #plot(fm_array, np.ones(5)*deg_dec_wind_sup,'+--k',label='Ceiling decision window BIPD')
    #plot(fm_array, np.ones(5)*deg_dec_wind_inf,'*--k',label='Floor decision window BIPD')
    errorbar(fm_array, [37, 40, 62, 83, 115],yerr=[46, 29, 29, 31,37], fmt='--r', label='Data')
    legend(loc='bottom right')
    grid()
    ylim(0, 180)
    xlabel('Modulation frequency (Hz)')
    ylabel('Extracted phase (deg)')
    
    #show()
    tight_layout()
    
    end = time.time()
    print 'Time elapsed: '+ str(end - start)


# +
def comparison_decoder(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi,Nx,Ny,fc,num_training,m,tc,sigma,k,poisson_noise,similarity_method,decision_window,width_frame,index_mf):
    simple_browsing(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi,Nx,Ny,fc,num_training,m,tc,sigma,k,poisson_noise,similarity_method,decision_window,width_frame,index_mf)
    #minimal_processing(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi,Nx,Ny,fc,num_training,tc,sigma,k, similarity_method,decision_window,width_frame)
    
       
tau=widgets.FloatSlider(min=1, max=10, step=0.5, value=6,description='Time constant exp decay kernel (width kernel)')
spread=widgets.FloatSlider(min=0.05,max=1,step=0.05,value=0.4,description='height of the kernel')
stre_inh=widgets.FloatSlider(min=0,max=10,step=0.1,value=2.6,description='Strength of inhibition')
kernel_func=widgets.Dropdown(options=['Centrally weighted exp', 'Border weighted exp', 'Constant','Identity'],value='Constant',description='Connectivity')
sig_con=widgets.FloatSlider(min=0.1,max=5,step=0.1,value=1,description='Sigma of the connectivity function')
mcAlp_bipd=widgets.Checkbox(value=True,description='McAlpine distribution BIPD',disabled=False)
dphi=widgets.FloatSlider(min=-1, max=1, step=1.0/100, value=0,description='Phase difference at t=0')
Nx=widgets.IntSlider(min=100, max=5000, step=100, value=250,description='Number of pixel on the time axis')
Ny=widgets.IntSlider(min=100, max=5000, step=100, value=100,description='Number of pixel on the BIPD axis')
fc=widgets.IntSlider(min=0, max=1000, step=100, value=500,description='Carrier frequency (Hz)')
num_training=widgets.IntSlider(min=10, max=1000, step=10, value=20,description='Number of training samples')
m=widgets.FloatSlider(min=0.1, max=1, step=0.05, value=1,description='Modulation depth')
tc=widgets.Dropdown(options=['Cosine with exponent', 'Exponential'],value='Cosine with exponent',description='Tuning Curve')
sigma=widgets.FloatSlider(min=0,max=2,step=0.1,value=0.8,description='Width tuning curve when choosing "Exponential"')
k=widgets.IntSlider(min=0, max=50, step=2, value=4,description='Exponent cosine k')
poisson_noise=widgets.Checkbox(value=False,description='Poisson noise',disabled=False)
similarity_method=widgets.Dropdown(options=['Euclidean distance','Summing and dot product','2D dot product'],value='2D dot product',description='Similarity distance between the 2 decision windows')
decision_window=widgets.FloatRangeSlider(value=[0,4*pi],min=0,max=4*pi,step=4*pi/1000,description='Decision window')
width_frame=widgets.FloatSlider(min=0.002, max=1, step=1/1000.0, value=0.005,description='Frame width ')
index_mf=widgets.IntSlider(min=0, max=4, step=1, value=0,description='Index modulation frequency to display') 
                         

s={'tau': tau,'spread':spread,'stre_inh':stre_inh,'kernel_func':kernel_func,'sig_con':sig_con,
                           'mcAlp_bipd':mcAlp_bipd,'dphi':dphi,'Nx':Nx,'Ny':Ny,'fc':fc,'num_training':num_training,
                           'm':m,'tc':tc,'sigma':sigma,'k':k,'poisson_noise':poisson_noise,'similarity_method':similarity_method,
                           'decision_window':decision_window,'width_frame':width_frame,'index_mf':index_mf}
def setting_up_tab(accordion):
    for i in xrange(len(accordion.children)):
        accordion.set_title(i,accordion.children[i].description)
        accordion.children[i].layout.width = '100%'
        accordion.children[i].style = {'description_width': '30%'}
        accordion.children[i].continuous_update = False 
                
kern_inh_acc = widgets.Accordion(children=[tau,spread,stre_inh,kernel_func,sig_con])
setting_up_tab(kern_inh_acc)

tuning_curve_acc = widgets.Accordion(children=[tc,sigma,k])
setting_up_tab(tuning_curve_acc)

dens_plot_acc = widgets.Accordion(children=[mcAlp_bipd,dphi,Nx,Ny,fc,num_training,m])
setting_up_tab(dens_plot_acc)
    
dec_making_acc=widgets.Accordion(children=[poisson_noise,similarity_method,decision_window,width_frame,index_mf])
setting_up_tab(dec_making_acc)
    
tab_contents = ['Density plot and training', 'Tuning curve', 'Kernel inhibition', 'Decision making']    
tab=widgets.Tab()    
children=[dens_plot_acc,tuning_curve_acc,kern_inh_acc,dec_making_acc]
tab.children=children
for i in range(len(children)):
    tab.set_title(i,tab_contents[i])
    
w=widgets.interactive_output(comparison_decoder,s)   
display(tab,w)


# -

def minimal_processing(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi,Nx,Ny,fc,num_training,tc,sigma,k, similarity_method,decision_window,width_frame):
    n=2
    exp_data=[37, 40, 62, 83, 115]
    num_fm=5
    dphi_pix=int(dphi*Ny)
    dphi_phase=dphi_pix*360.0/Ny
      
    training_patts=training(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fc,num_training,tc,sigma,k)
    testing_patts=testing_dyn_ipd(tau,spread,stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi_phase,Nx,Ny,fc,tc,sigma,k)
    
    decision_window_pix_inf=int(decision_window[0]*n*Nx/(2*pi*n))
    deg_dec_wind_inf=(decision_window[0]%(2*pi))*(180.0)/pi + dphi_phase
    deg_dec_wind_inf%=360
    deg_dec_wind_sup=(decision_window[1]%(2*pi))*(180.0)/pi + dphi_phase
    deg_dec_wind_sup%=360
    
    decision_window_pix_sup=int(decision_window[1]*n*Nx/(2*pi*n))
    decision_window_pix=[decision_window_pix_inf, decision_window_pix_sup] 
    
    res=testing(dphi_pix,Ny,num_training,testing_patts,training_patts,similarity_method,decision_window_pix,decision_window_pix[1]-decision_window_pix[0])
    width_frame_pix=int(width_frame*Nx)


    BIPD_vs_time=np.zeros((num_fm,num_training,Nx))
    for i in xrange(Nx):
            BIPD_vs_time[:,:,i]=testing(dphi_pix,Ny,num_training,testing_patts,training_patts,similarity_method,
                                        [Nx-width_frame_pix+i,Nx+i],width_frame_pix)[0]
            
    max_simi=np.zeros((num_fm,Nx))
    arg_max_simi=np.zeros((num_fm,Nx))
    std_circ=np.zeros((num_fm,Nx))
    mean_circ=np.zeros((num_fm,Nx))
    arg_mm=np.zeros(num_fm)
    time_max=np.zeros(num_fm)
    sum_sin=np.zeros((num_fm,Nx))
    sum_cos=np.zeros((num_fm,Nx))
    ipds=np.linspace(0,2*pi,num_training,endpoint=False)
    for k in xrange(num_fm):
        for i in xrange(Nx):
            for j in xrange(num_training):
                sum_sin[k,i]+=np.sin(ipds[j]+0*dphi_phase*pi/180.0)*BIPD_vs_time[k][j,i]
                sum_cos[k,i]+=np.cos(ipds[j]+0*dphi_phase*pi/180.0)*BIPD_vs_time[k][j,i]
            mean_circ[k,i]=(arctan2(sum_sin[k,i],sum_cos[k,i])*180.0/pi)
            std_circ[k,i]=sqrt(-np.log((1.0/sum(BIPD_vs_time[k][:,i])**2)*(sum_sin[k,i]**2+sum_cos[k,i]**2)))*180.0/pi
            max_simi[k,i]=np.amax(BIPD_vs_time[k][:,i])*180
            arg_max_simi[k,i]=np.argmax(BIPD_vs_time[k][:,i])    
            if mean_circ[k,i]<0:
                mean_circ[k,i]=360+mean_circ[k,i]

        time_max[k]=argmax(max_simi[k,:])
     
        arg_mm[k]=arg_max_simi[k,int(time_max[k])]     
    arg_mm=arg_mm*360.0/(num_training)
    #print arg_mm.shape
    #print res[1].shape
    barg=arg_mm>0.0
    bres=res[1>0.0]
    
    mse_arg=np.mean((arg_mm-exp_data)**2)
    mse_res=np.mean((res[1]-exp_data)**2)
    #print mse_arg,mse_res
    
    if not np.any(arg_mm):
        mse_arg=-1
    if not np.any(res[1]):
        mse_res=-1
    
    
    return mse_arg,mse_res
    

def param_map_kernel_shape(height_vec,width_vec):
    stre_inh=3.0
    kernel_fun='Constant'
    sig_con=1.0
    mcAlp_bipd=True
    dphi=0
    Nx=500
    Ny=100
    fc=500
    num_training=20
    tc='Cosine with exponent'
    sigma=0.8
    k=4
    similarity_method='2D dot product'
    decision_window=[0,999]
    width_frame=0.002
    res=np.zeros((2,len(height_vec),len(width_vec)))
    cmpt=0
    for i in xrange(len(height_vec)):
        for j in xrange(len(width_vec)):
            res[:,i,j]=minimal_processing(width_vec[j],height_vec[i],stre_inh,kernel_func,sig_con,mcAlp_bipd,dphi,Nx,Ny,fc,num_training,tc,sigma,k,similarity_method,decision_window,width_frame)
            #print res
            cmpt=cmpt+1
            print cmpt
    save('param_map_kernel_shape',res)    
    return res

# +
fig=figure(figsize=(10,4))
subplot(121)
title('Argmax')
imshow(sqrt(res[0,:]),origin='lower_left', interpolation=None,aspect='auto',extent=[1,10,0.1,1],vmin=0,vmax=180)

xlabel('Width kernel')
ylabel('Height kernel')
colorbar()
subplot(122)
title('2D dot product')imshow(sqrt(res[1,:]),origin='lower_left', interpolation=None,aspect='auto',extent=[1,10,0.1,1],vmin=0,vmax=180)
xlabel('Width kernel')
ylabel('Height kernel')
colorbar()
fig.tight_layout()
# -


