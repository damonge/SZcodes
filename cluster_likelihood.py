import numpy as np
import py_cosmo_mad as csm
from parameters import *
from scipy.interpolate import interp1d
import profiles as prf
import matplotlib.pyplot as plt
import integrals as intg
import cosmo as cos
import time
from scipy.special import erf

DEBUG_ERROR_FAC=1.

ISQ2=1./np.sqrt(2.)

def plotarr(array,title,x_arr,y_arr,
            vmin=1234,vmax=1234,
            figname='none',x_label='x',y_label='y') :
    ext=[x_arr[0],x_arr[-1],y_arr[0],y_arr[-1]]
    plt.figure()
    plt.title(title,fontsize=16)
    if vmin==1234 : vmn=np.amin(array)
    else : vmn=vmin
    if vmax==1234 : vmx=np.amax(array)
    else : vmx=vmax
    plt.imshow(array,interpolation='none',origin='lower',extent=ext,
               vmin=vmn,vmax=vmx,aspect='auto')
    plt.xlabel(x_label,fontsize=16)
    plt.ylabel(y_label,fontsize=16)
    plt.colorbar()

def prob_mass(m_hi,m_lo,m_mean,sigma_m) :
#    x=(kappa-kappa_mean)/sigma_kappa
#    return np.log(10.)*kappa*np.exp(-0.5*x**2)/(sigma_kappa*np.sqrt(2*np.pi))
    x_hi=ISQ2*(m_hi-m_mean)/sigma_m
    x_lo=ISQ2*(m_lo-m_mean)/sigma_m
    return 0.5*(erf(x_hi)-erf(x_lo))

def get_cluster_distribution_qkz(cospar,exper,lth_arr,cov_arr,add_kappa=True) :
    fsky=exper['fsky']
    nz=exper['nz']
    z_min=exper['z_min']
    z_max=exper['z_max']
    nm=exper['nm']
    lm_min=exper['lm_min']
    lm_max=exper['lm_max']
    nq=exper['nq']
    lq_min=np.log(exper['q_min'])
    lq_max=np.log(exper['q_max'])
    if add_kappa :
        nk=exper['nk']
    dz=(z_max-z_min)/nz
    dlm=(lm_max-lm_min)/nm
    dlq=(lq_max-lq_min)/nq

    #Setup axes
    q_arr=np.exp(lq_min+(lq_max-lq_min)*(np.arange(nq)+0.5)/nq)
    m_arr  =10.**(lm_min+(lm_max-lm_min)*(np.arange(nm)+0.5)/nm)
    if add_kappa :
        mlo_arr=10.**(lm_min+(lm_max-lm_min)*(np.arange(nk)+0.0)/nk)
        mhi_arr=10.**(lm_min+(lm_max-lm_min)*(np.arange(nk)+1.0)/nk)
    z_arr=z_min+(z_max-z_min)*(np.arange(nz)+0.5)/nz

    #Set mass function
    #TODO: compute mass function directly fron CosmoMAD
    pcs,nm_arr=cos.get_mfunc(cospar,z_min,z_max,nz,lm_min,lm_max,nm,fsky)

    #Set up SZ noise
    sigY_1d=np.sqrt(cov_arr[0,0])
    sigYi=interp1d(lth_arr,sigY_1d,kind='cubic',bounds_error=False,fill_value=0)
    def sigY_f(th) :
        shp=np.shape(th)
        thb=th.flatten()
        lth=np.log10(thb)
        ind_below=np.where(lth<lth_arr[0])
        ind_above=np.where(lth>lth_arr[-1])
        ind_between=np.where((lth>=lth_arr[0]) & (lth<=lth_arr[-1]))
        sigout=np.zeros_like(thb)
        sigout[ind_below]=sigY_1d[0]*10.**(np.log10(sigY_1d[1]/sigY_1d[0])*(lth[ind_below]-lth_arr[0])/
                                           (lth_arr[1]-lth_arr[0]))
        sigout[ind_above]=sigY_1d[-1]*10.**(np.log10(sigY_1d[-1]/sigY_1d[-2])*(lth[ind_above]-lth_arr[-1])/
                                            (lth_arr[-1]-lth_arr[-2]))
        sigout[ind_between]=sigYi(lth[ind_between])
        return np.reshape(sigout,shp)

    #Set up kappa noise
    if add_kappa :
        sigk_1d=np.sqrt(cov_arr[2,2])/DEBUG_ERROR_FAC
        sigki=interp1d(lth_arr,sigk_1d,kind='cubic',bounds_error=False,fill_value=0)
        def sigk_f(th) :
            shp=np.shape(th)
            thb=th.flatten()
            lth=np.log10(thb)
            ind_below=np.where(lth<lth_arr[0])
            ind_above=np.where(lth>lth_arr[-1])
            ind_between=np.where((lth>=lth_arr[0]) & (lth<=lth_arr[-1]))
            sigout=np.zeros_like(thb)
            sigout[ind_below]=sigk_1d[0]*10.**(np.log10(sigk_1d[1]/sigk_1d[0])*(lth[ind_below]-lth_arr[0])/
                                               (lth_arr[1]-lth_arr[0]))
            sigout[ind_above]=sigk_1d[-1]*10.**(np.log10(sigk_1d[-1]/sigk_1d[-2])*(lth[ind_above]-lth_arr[-1])/
                                                (lth_arr[-1]-lth_arr[-2]))
            sigout[ind_between]=sigki(lth[ind_between])
            return np.reshape(sigout,shp)

    #Set signal
    y500_arr=np.zeros([nz,nm])
    th500_arr=np.zeros([nz,nm])
    if add_kappa :
        m500_arr=np.zeros([nz,nm])
        prefac_kappa=np.zeros(nz)
    for iz in np.arange(nz) :
        z=z_arr[iz]
        y500_arr[iz,:]=prf.y_500(m_arr,z,cospar,pcs)
        th500_arr[iz,:]=prf.theta_500(m_arr,z,pcs)
        if add_kappa :
            prefac_kappa[iz]=1E14/prf.kappa_prefactor(z,pcs)
            m500_arr[iz,:]=prf.kappa_500(m_arr,z,cospar,pcs)*prefac_kappa[iz]

    #Set noise
    sigY_arr=sigY_f(th500_arr)
    if add_kappa :
        sigk_arr=sigk_f(th500_arr)
        sigm_arr=sigk_arr*prefac_kappa[:,None]

    #Compute completeness
    if add_kappa :
        pdf_mass_arr=np.zeros([2*nk+1,nz,nm])
        pdf_mass_arr[nk+1:,:,:]=prob_mass(mhi_arr[:,None,None],mlo_arr[:,None,None],m500_arr[None,:,:],sigm_arr[None,:,:])
        pdf_mass_arr[:nk,:,:]=prob_mass(-mlo_arr[:,None,None],-mhi_arr[:,None,None],m500_arr[None,:,:],sigm_arr[None,:,:])
        pdf_mass_arr[nk,:,:]=prob_mass(mlo_arr[0,None,None],-mlo_arr[0,None,None],m500_arr[None,:,:],sigm_arr[None,:,:])
    pdf_qSZ_arr=intg.integ_pdf_array(y500_arr,sigY_arr,q_arr,cospar['sigma_lny'])

    if DEBUG :
        lm_arr=np.log10(m_arr)
        plotarr(np.log10(m500_arr),"m500",lm_arr,z_arr)
        plotarr(np.log10(sigm_arr),"s_m500",lm_arr,z_arr)
        plotarr(m500_arr/sigm_arr,"S/N m",lm_arr,z_arr)
        plt.show()
        for i in np.arange(2*nk+1) :
            plotarr(pdf_mass_arr[i,:,:],"%d"%i,lm_arr,z_arr)
            if i%16==0 :
                plt.show()

    if add_kappa :
        n_arr=np.sum(nm_arr[None,None,:,:]*pdf_qSZ_arr[None,:,:,:]*pdf_mass_arr[:,None,:,:],axis=3)*dlm*dz*dlq
    else :
        n_arr=np.sum(nm_arr[None,:,:]*pdf_qSZ_arr[:,:,:],axis=2)*dlm*dz*dlq

    n_persqamin=(np.sum(n_arr)/(4*np.pi*exper['fsky']*(180*60/np.pi)**2))
    chi_arr=np.sum(pdf_qSZ_arr,axis=0)
    th500_mean=np.sum(chi_arr*nm_arr*th500_arr)/np.sum(chi_arr*nm_arr)
    theta_hard=2*max([exper['beam_reduced'],th500_mean])/2.355
    blendedfrac=1-np.exp(-n_persqamin*np.pi*(2*theta_hard)**2)

#    lm_arr=np.log10(m_arr)
#    print "Average cluster size: %lf"%th500_mean
#    print "Number of clusters : %d"%(int(np.sum(n_arr)))
#    print "Blended fraction : %lE"%blendedfrac
#    plotarr(np.log10(chi_arr*nm_arr),"$dN/dzd\\log_{10}M$",lm_arr,z_arr,x_label="$\\log_10 M/(M_\\odot/h)$",
#            y_label="$z$",vmin=0); plt.show()

    return n_arr*(1-blendedfrac)

def get_cluster_distribution_mz(cospar,exper,lth_arr,cov_arr) :
    fsky=exper['fsky']
    nz=exper['nz']
    z_min=exper['z_min']
    z_max=exper['z_max']
    nm=exper['nm']
    lm_min=exper['lm_min']
    lm_max=exper['lm_max']
    nq=exper['nq']
    lq_min=np.log(exper['q_min'])
    lq_max=np.log(exper['q_max'])
    dz=(z_max-z_min)/nz
    dlm=(lm_max-lm_min)/nm
    dlq=(lq_max-lq_min)/nq

    #Setup axes
    m_arr  =10.**(lm_min+(lm_max-lm_min)*(np.arange(nm)+0.5)/nm)
    z_arr=z_min+(z_max-z_min)*(np.arange(nz)+0.5)/nz
    q_arr=np.exp(lq_min+(lq_max-lq_min)*(np.arange(nq)+0.5)/nq)

    #Set mass function
    #TODO: compute mass function directly fron CosmoMAD
    pcs,nm_arr=cos.get_mfunc(cospar,z_min,z_max,nz,lm_min,lm_max,nm,fsky)

    #Set up SZ noise
    sigY_1d=np.sqrt(cov_arr[0,0])
    sigYi=interp1d(lth_arr,sigY_1d,kind='cubic',bounds_error=False,fill_value=0)
    def sigY_f(th) :
        shp=np.shape(th)
        thb=th.flatten()
        lth=np.log10(thb)
        ind_below=np.where(lth<lth_arr[0])
        ind_above=np.where(lth>lth_arr[-1])
        ind_between=np.where((lth>=lth_arr[0]) & (lth<=lth_arr[-1]))
        sigout=np.zeros_like(thb)
        sigout[ind_below]=sigY_1d[0]*10.**(np.log10(sigY_1d[1]/sigY_1d[0])*(lth[ind_below]-lth_arr[0])/
                                           (lth_arr[1]-lth_arr[0]))
        sigout[ind_above]=sigY_1d[-1]*10.**(np.log10(sigY_1d[-1]/sigY_1d[-2])*(lth[ind_above]-lth_arr[-1])/
                                            (lth_arr[-1]-lth_arr[-2]))
        sigout[ind_between]=sigYi(lth[ind_between])
        return np.reshape(sigout,shp)

    #Set signal
    y500_arr=np.zeros([nz,nm])
    th500_arr=np.zeros([nz,nm])
    for iz in np.arange(nz) :
        z=z_arr[iz]
        y500_arr[iz,:]=prf.y_500(m_arr,z,cospar,pcs)
        th500_arr[iz,:]=prf.theta_500(m_arr,z,pcs)

    #Set noise
    sigY_arr=sigY_f(th500_arr)
    pdf_qSZ_arr=intg.integ_pdf_array(y500_arr,sigY_arr,q_arr,cospar['sigma_lny'])

    chi_arr=np.sum(pdf_qSZ_arr,axis=0)*dlq
    n_persqamin=(np.sum(nm_arr*chi_arr*dlm*dz)/(4*np.pi*exper['fsky']*(180*60/np.pi)**2))
    th500_mean=np.sum(chi_arr*nm_arr*th500_arr)/np.sum(chi_arr*nm_arr)
    theta_hard=2*max([exper['beam_reduced'],th500_mean])/2.355
    blendedfrac=1-np.exp(-n_persqamin*np.pi*(2*theta_hard)**2)

    print "Average cluster size: %lf"%th500_mean
    print "Number of clusters : %d"%(int(np.sum(nm_arr*chi_arr*dz*dlm)))
    print "Blended fraction : %lE"%blendedfrac
    lm_arr=np.log10(m_arr)
    plotarr(np.log10(chi_arr*nm_arr),"$dN/dzd\\log_{10}M$",lm_arr,z_arr,x_label="$\\log_10 M/(M_\\odot/h)$",
            y_label="$z$",vmin=0); plt.show()

    return z_arr,lm_arr,chi_arr*nm_arr*(1-blendedfrac)
