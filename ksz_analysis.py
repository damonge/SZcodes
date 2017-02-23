import numpy as np
import matplotlib.pyplot as plt
import profiles as prf
import experiments as xpr
import cluster_likelihood as cll
import cosmo as cos
import os
from scipy.interpolate import interp1d
from scipy.integrate import quad
from parameters import *

VR_E0=0.503807
VR_AE=-0.01010767
VR_K0=-0.02195494
VR_AK=-1.88330603
def epsilon_beta(m,z,whichexp) :
    if whichexp=='ideal' :
        prefac=1.
    elif whichexp=='desi' :
        prefac=2.
    elif whichexp=='boss' :
        prefac=3.
    else :
        exit(1)

    return VR_E0*prefac

def mk_prediction(xp,specname) :
    lk,pk,lx,px=prf.compute_fourier_profiles()
    lth_arr,cov_arr=prf.get_covariance_matched_filter(xp,"bode_almost_wmap5_lmax_1e4","All")
    cpar0={'och2':0.1197,'obh2':0.02222,'hh':0.69,'w0':-1.0,'wa':0.00,
           'tau':0.06,'mnu':60.,'a_s':2.1955,'ns':0.9655,'pan':0.000,'mfName':'Tinker500',
           'hbias':0.800,'Ay':3.350,'alphay':1.79,'betay':0.00,'gammay':0.00,'sigma_lny':0.127}
    
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
    
    sigK_1d=np.sqrt(cov_arr[1,1])
    sigKi=interp1d(lth_arr,sigK_1d,kind='cubic',bounds_error=False,fill_value=0)
    def sigK_f(th) :
        shp=np.shape(th)
        thb=th.flatten()
        lth=np.log10(thb)
        ind_below=np.where(lth<lth_arr[0])
        ind_above=np.where(lth>lth_arr[-1])
        ind_between=np.where((lth>=lth_arr[0]) & (lth<=lth_arr[-1]))
        sigout=np.zeros_like(thb)
        sigout[ind_below]=sigK_1d[0]*10.**(np.log10(sigK_1d[1]/sigK_1d[0])*(lth[ind_below]-lth_arr[0])/
                                           (lth_arr[1]-lth_arr[0]))
        sigout[ind_above]=sigK_1d[-1]*10.**(np.log10(sigK_1d[-1]/sigK_1d[-2])*(lth[ind_above]-lth_arr[-1])/
                                            (lth_arr[-1]-lth_arr[-2]))
        sigout[ind_between]=sigKi(lth[ind_between])
        return np.reshape(sigout,shp)

    z_arr,lm_arr,nmz_arr,pcs=cll.get_cluster_distribution_mz(cpar0,xp,lth_arr,cov_arr)
    print np.sum((z_arr[1]-z_arr[0])*(lm_arr[1]-lm_arr[0])*nmz_arr)

    y500_arr=np.array([[prf.y_500(m,z,cpar0,pcs) for m in 10.**lm_arr] for z in z_arr])
    tau500_arr=np.array([[prf.tau_500(m,z,cpar0,pcs) for m in 10.**lm_arr] for z in z_arr])
    th500_arr=np.array([[prf.theta_500(m,z,pcs) for m in 10.**lm_arr] for z in z_arr])
    sigY_arr=sigY_f(th500_arr)
    sigK_arr=sigK_f(th500_arr)

    inv_epsilon_arr=np.zeros_like(nmz_arr)
    eps_cmb_arr=np.zeros_like(nmz_arr)
    for iz in np.arange(len(z_arr)) :
        z=z_arr[iz]
        for im in np.arange(len(lm_arr)) :
            m=10.**lm_arr[im]
            sv=prf.sigma_v(m,z)
            eps_beta=epsilon_beta(m,z,specname)
            eps_tau=np.sqrt((0.41*sigY_arr[iz,im]/y500_arr[iz,im])**2+0.15**2)
            eps_cmb0=sigK_arr[iz,im]/tau500_arr[iz,im]
            eps_cmb_arr[iz,im]=eps_cmb0*CLIGHT/sv
            
            def integ_beta(v) :
                beta=v/CLIGHT
                eps_cmb=eps_cmb0/beta
                return np.exp(-0.5*(v/sv)**2)/((eps_beta**2+eps_tau**2+eps_cmb**2+
                                                (eps_beta*eps_tau)**2)*np.sqrt(2*np.pi*sv**2))

            inv_epsilon_arr[iz,im]=2*quad(integ_beta,0.001*sv,10*sv)[0]

#    cll.plotarr(1./eps_cmb_arr,"SN",lm_arr,z_arr,vmin=0,
#                x_label="$\\log_{10} M/(M_\\odot/h)$",y_label="$z$")
    cll.plotarr(np.log10(nmz_arr),"N",lm_arr,z_arr,vmin=0,
                x_label="$\\log_{10} M/(M_\\odot/h)$",y_label="$z$")
#    cll.plotarr(inv_epsilon_arr,"1/E",lm_arr,z_arr,vmin=0,
#                x_label="$\\log_{10} M/(M_\\odot/h)$",y_label="$z$")
#    cll.plotarr(np.log10(nmz_arr*inv_epsilon_arr),"N/E",lm_arr,z_arr,vmin=0,
#                x_label="$\\log_{10} M/(M_\\odot/h)$",y_label="$z$")

    sigm2_alpha_arr=np.sum(nmz_arr*inv_epsilon_arr*(lm_arr[1]-lm_arr[0]),axis=1)
    sigm2_alpha_f=interp1d(z_arr,sigm2_alpha_arr,kind='cubic',bounds_error=False,fill_value=0)

    nz=20
    z0_arr=0.1*np.arange(nz)
    zf_arr=0.1*np.arange(nz)+0.1
    sigma_alpha_arr=np.array([1./np.sqrt(quad(sigm2_alpha_f,z0_arr[iz],zf_arr[iz])[0]) for iz in np.arange(nz)])
    
    np.savetxt(xp['name']+'_'+specname+'_sigma_alpha.txt',
               np.transpose([0.5*(z0_arr+zf_arr),sigma_alpha_arr]))

    plt.figure()
    plt.plot(0.5*(z0_arr+zf_arr),sigma_alpha_arr)
    plt.show()

for xp in [xpr.exp_s3,xpr.exp_s4b3c] :
    for specname in ['ideal','desi','boss'] :
        mk_prediction(xp,specname)
