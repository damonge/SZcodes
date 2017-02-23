import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sint
from scipy.integrate import quad, fixed_quad
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as spline
from scipy.special import jv,erf
from scipy.optimize import fsolve
import py_cosmo_mad as csm
import os
import integrals as intg
from experiments import *
from parameters import *

XZMIN=1E-4
XZMAX=100
LX_MIN=-3.
LX_MAX=3.
NX=128
LK_MIN=-3.
LK_MAX=2.
NK=128
P_CONC=1.18
P_ALPHA=1.05
P_BETA=5.49
P_GAMMA=0.308
P_NORM_RATIO=2.91036411996
NTH=64
LTH_MIN=-3.
LTH_MAX=2.

#Frequency dependence of the profile
def fnu_sz(nu,component) : #tSZ
    if component==0 :
        x=0.017611907*nu
        ex=np.exp(x)
        return x*(ex+1)/(ex-1)-4
    else : #kSZ
        return np.ones_like(nu)

def sigma_v(m,z) :
    return 312.0*(1+z)**0.87+22.0*(1+z)**1.05*np.log10(m*1E-14)

#Virial radius
def r_500(m,z,pcs) :
    rhocrit=RHOCRIT0*(pcs.hubble(1./(1+z))/pcs.hubble(1.))**2
    return (3*m/(4*500*np.pi*rhocrit))**0.333333333

#Projected virial radius
def theta_500(m,z,pcs) :
    da=pcs.angular_diameter_distance(1./(1+z))
    r=r_500(m,z,pcs)
    th1=RAD2AMIN*r/da
    return th1

#tSZ amplitude
def y_500(m,z,pars,pcs) :
    da=pcs.angular_diameter_distance(1./(1+z))
    ez=pcs.hubble(1./(1+z))/pcs.hubble(1.)
    prefac=1E-10*(100/da)**2*ez**0.6667
    logm=np.log(pars['hbias']*m/1.5E14)
    rescale=1.49
#    return prefac*pars['Ay']*(0.8*m*1E-14)**pars['alphay']
    return prefac*pars['Ay']*rescale*np.exp(logm*pars['alphay']+logm**2*pars['betay'])*(1+z)**pars['gammay']

#kSZ amplitude
def tau_500(m,z,pars,pcs) :
    y5=y_500(m,z,pars,pcs)
    r5=r_500(m,z,pcs)
    m5=m*1E-14
    norm_profiles=P_NORM_RATIO
    return 193.0*norm_profiles*r5*y5/m5

def kappa_prefactor(z,pcs) :
    d_clu=pcs.angular_diameter_distance(1./(1+z))
    d_cmb=pcs.angular_diameter_distance(1./(1+ZCMB))
    d_clu_cmb=(pcs.radial_comoving_distance(1./(1+ZCMB))-pcs.radial_comoving_distance(1./(1+z)))/(1+ZCMB)
    return 7.55919942E-4*d_clu_cmb/(d_clu*d_cmb)/(4*np.pi)

#kappa amplitude
def kappa_500(m,z,pars,pcs) :
    m5=m*1E-14
    return m5*kappa_prefactor(z,pcs)
#    chi_clu=pcs.radial_comoving_distance(1./(1+z))
#    chi_cmb=pcs.radial_comoving_distance(1./(1+ZCMB))
#    return 6.01541976E-5*(chi_cmb-chi_clu)/(chi_clu*chi_cmb)*(1+z)**2*m5

#Pressure profile
def p_gnfw(rx) :
    cx=P_CONC*rx
    return 1./(cx**P_GAMMA*(1+cx**P_ALPHA)**((P_BETA-P_GAMMA)/P_ALPHA))

#Density profile
def p_dnfw(rx) :
    cx=P_CONC*rx
    return 1./(cx*(1+cx)**2)

#Cumulative matter profile
def p_mnfw(rx) :
    cx=P_CONC*rx
    prefac=(1+P_CONC)/((1+P_CONC)*np.log(1+P_CONC)-P_CONC)
    postfac=((1+cx)*np.log(1+cx)-cx)/(1+cx)
    return prefac*postfac

#Electron number density profile
def p_nnfw(rx) :
    cx=P_CONC*rx
    pp=p_gnfw(rx)
    pm=p_mnfw(rx)
    x2prime=pp*rx*(P_GAMMA+P_BETA*cx**P_ALPHA)/(1+cx**P_ALPHA)

    return x2prime/pm

#Wrapper for all cluster profiles                  
def profile_real(z,x,comp) :
    rx=np.sqrt(z*z+x*x)
    if comp==0 : #tSZ
        return p_gnfw(rx)
    elif comp==1 : #kSZ
        return p_nnfw(rx)
    elif comp==2 : #kappa
        return p_dnfw(rx)

#Projected profile
def profile_project_d(x,comp) :
    return 2*quad(profile_real,XZMIN,XZMAX,args=(x,comp))[0]

#Spherical normalization
def norm_sphere(comp,nr) :
    def integ_sphere(r) :
        return r**2*profile_real(0,r,comp)
    return quad(integ_sphere,10**LX_MIN,nr)[0]

def norm_cylind(comp,nr) :
    def integ_cylind(r) :
        return 0.5*r*profile_project_d(r,comp)
    return quad(integ_cylind,10**LX_MIN,nr)[0]

#Compute arrays of Fourier-space profiles
def compute_fourier_profiles() :
    prefix="save"
    if os.path.isfile(prefix+"_profiles_fourier.txt") :
        lk_arr,pk_0_arr,pk_1_arr,pk_2_arr=np.loadtxt(prefix+"_profiles_fourier.txt",unpack=True)
        k_arr=10.**lk_arr
        pk_arr=np.array([pk_0_arr,pk_1_arr,pk_2_arr])
        lx_arr,px_0_arr,px_1_arr,px_2_arr=np.loadtxt(prefix+"_profiles_real.txt",unpack=True)
        px_arr=np.array([px_0_arr,px_1_arr,px_2_arr])
        x_arr=10.**lx_arr
    else :
        norms=np.array([norm_sphere(c,1.) for c in np.arange(3)])
        lx_arr=LX_MIN+(LX_MAX-LX_MIN)*(np.arange(NX)+0.5)/NX
        x_arr=10.**lx_arr
        px_arr=np.array([[profile_project_d(x,c)/norms[c] for x in x_arr] for c in np.arange(3)])
        np.savetxt(prefix+"_profiles_real.txt",np.transpose([lx_arr,px_arr[0,:],px_arr[1,:],px_arr[2,:]]))

        profile_project_i=[]
        for i in np.arange(3) :
            profile_project_i.append(interp1d(lx_arr,px_arr[i,:],
                                              kind='cubic',bounds_error=False,fill_value=0))

        def profile_project_f(x,comp) :
            lx=np.log10(x)
            return (profile_project_i[comp])(lx)

        def p_project_fourier_d(k,comp) :
            def integ_fourier(lx) :
                x=10.**lx
                return x*x*profile_project_f(x,comp)*jv(0,k*x)
            return np.log(10.)*quad(integ_fourier,LX_MIN,LX_MAX,limit=3000)[0]
#            def integ_fourier(x) :
#                return x*profile_project_f(x,comp)*jv(0,k*x)
#            return quad(integ_fourier,10**LX_MIN,10**LX_MAX,limit=1000)[0]

        lk_arr=LK_MIN+(LK_MAX-LK_MIN)*(np.arange(NK)+0.5)/NK
        k_arr=10.**lk_arr
        pk_arr=np.array([[p_project_fourier_d(k,c) for k in k_arr] for c in np.arange(3)])
        np.savetxt(prefix+"_profiles_fourier.txt",np.transpose([lk_arr,pk_arr[0,:],pk_arr[1,:],pk_arr[2,:]]))

    return lk_arr,pk_arr,lx_arr,px_arr

#Read all relevant CMB power spectra from CAMB files
def read_cmb_powerspec(prefix_camb):
    cl_fid={}
    l,cl_fid['TT'],cl_fid['EE'],cl_fid['BB'],cl_fid['TE']=np.loadtxt(prefix_camb+'_lensedCls.dat',unpack=True)
    l,cl_fid['TT_unlen'],cl_fid['EE_unlen'],cl_fid['TE_unlen'],cl_fid['kappa'],cl_fid['Tphi']=np.loadtxt(prefix_camb+'_scalCls.dat',unpack=True)
    f=(2*np.pi)/((TCMB*1E6)**2*(l*(l+1)))
    cl_fid['TT'],cl_fid['EE'],cl_fid['BB'],cl_fid['TE']=cl_fid['TT']*f,cl_fid['EE']*f,cl_fid['BB']*f,cl_fid['TE']*f
    cl_fid['TT_unlen'],cl_fid['EE_unlen'],cl_fid['TE_unlen']=cl_fid['TT_unlen']*f,cl_fid['EE_unlen']*f,cl_fid['TE_unlen']*f
    
#    cl_fid['kappa']=cl_fid['kappa']/(l**4*(TCMB*1E6)**2)
    cl_fid['kappa']=(1+1./l)**2*cl_fid['kappa']/(TCMB*1E6)**2/4
    
    cl_fid['BB_unlen']=cl_fid['TT_unlen']*0
    return(l,cl_fid)

#Set noise powerspectrum for cluster lensing
def set_kappa_noisepower(prefix_camb,exp,flatsky) :
    ell,cl_fid=read_cmb_powerspec(prefix_camb)
    lmax=10000; lmin=2;
    if flatsky :
        print "Computing lensing noise power spectrum"
        deltaL=50
        sigma2_rad=(exp['noise_reduced']/(RAD2AMIN*TCMB*1E6))**2
        beam_rad=exp['beam_reduced']/(RAD2AMIN*2.355)
    
        nl={}
        nl['TT']=sigma2_rad*np.exp(ell*(ell+1.)*beam_rad**2)
        nl['EE']=2*sigma2_rad*np.exp(ell*(ell+1.)*beam_rad**2)
        nl['BB']=2*sigma2_rad*np.exp(ell*(ell+1.)*beam_rad**2)
        nl['TE']=0*nl['TT']
        
        l_lens_min_temp=2
        l_lens_min_pol=2
        l_lens_max_temp=10000
        l_lens_max_pol=10000
        
        lensingSpec=['TT','EE','BB']
        
        l1d=np.arange(-lmax,lmax,deltaL)
        lx,ly=np.meshgrid(l1d,l1d)
        l=np.sqrt(lx*lx+ly*ly)
        
        clunlenFunc={}
        clunlenArr={}
        cltotFunc={}
        cltotArr={}
        for key in lensingSpec :
            cl_tot=cl_fid[key]+nl[key]
            cl_unlen=cl_fid[key+'_unlen']
            clunlenFunc[key]=interp1d(ell,cl_unlen,kind='linear',bounds_error=False,fill_value=0.)
            clunlenArr[key]=clunlenFunc[key](l)
            if key=='TT' :
                cl_tot[np.where(ell<l_lens_min_temp)]*=1E6
                cl_tot[np.where(ell>l_lens_max_temp)]*=1E6
            if key=='EE' or key=='BB' :
                cl_tot[np.where(ell<l_lens_min_pol)]*=1E6
                cl_tot[np.where(ell>l_lens_max_pol)]*=1E6
            if key!='TE' :
                cltotFunc[key]=interp1d(ell,cl_tot,kind='linear',bounds_error=False,fill_value=1E50)
            else :
                cltotFunc[key]=interp1d(ell,cl_tot,kind='linear',bounds_error=False,fill_value=0.)
            cltotArr[key]=cltotFunc[key](l)

        n_Ls=200
        LogLs=np.linspace(np.log(2),np.log(lmax+0.1),n_Ls)
        Ls=np.unique(np.floor(np.exp(LogLs)).astype(int))

        cl1_TT_unlen=clunlenArr['TT']; cl1_TT_tot=cltotArr['TT']
        cl1_EE_unlen=clunlenArr['EE']; cl1_EE_tot=cltotArr['EE']
        cl1_BB_tot=cltotArr['BB']
        NL_TT=[]; NL_EE=[]; NL_EB=[]
        for L in Ls :
            l2=np.sqrt((L-lx)**2+ly**2)
            Ldotl1=L*lx
            Ldotl2=L*(L-lx)

            cosphi= np.nan_to_num((Ldotl1-l**2)/(l*l2))
            sinphi=-np.nan_to_num((L*ly)/(l*l2))
            sin2phi=2*cosphi*sinphi
            cos2phi=np.nan_to_num(2.*((Ldotl1-l**2)/(l*l2))**2-1.)
            sin2phiSq=4.*cosphi**2*(1-cosphi**2)

            cl2_TT_unlen=clunlenFunc['TT'](l2)
            cl2_TT_tot=cltotFunc['TT'](l2)
            cl2_EE_unlen=clunlenFunc['EE'](l2)
            cl2_EE_tot=cltotFunc['EE'](l2)
            cl2_BB_tot=cltotFunc['BB'](l2)
            
            kernel=(cl1_TT_unlen*Ldotl1+cl2_TT_unlen*Ldotl2)**2/(2*cl1_TT_tot*cl2_TT_tot)
            integral=np.sum(kernel)*(2*np.pi)**(-2.)*deltaL**2
            NL_TT.append((integral)**(-1))
            
            kernel=((cl1_EE_unlen*Ldotl1+cl2_EE_unlen*Ldotl2)*cos2phi)**2/(2*cl1_EE_tot*cl2_EE_tot)
            integral=np.sum(kernel)*(2*np.pi)**(-2.)*deltaL**2
            NL_EE.append((integral)**(-1))
            
            kernel=sin2phiSq*(cl1_EE_unlen*Ldotl1)**2/(cl1_EE_tot*cl2_BB_tot)
            integral=np.sum(kernel)*(2*np.pi)**(-2.)*deltaL**2
            NL_EB.append((integral)**(-1))

        NL_TT=(Ls*(Ls+1))**2*np.array(NL_TT)/4.
        s = spline(np.log(Ls), np.log(NL_TT), s=0)
        NL_TT=np.exp(s(np.log(ell)))

    #TODO: for some reason this is not logged
        NL_EE=(Ls*(Ls+1))**2*np.array(NL_EE)/4.
        s=spline(Ls,NL_EE,s=0)
        NL_EE=s(ell)

        NL_EB=(Ls*(Ls+1))**2*np.array(NL_EB)/4.
        s=spline(Ls,NL_EB,s=0)
        NL_EB=s(ell)
        ell1=ell
        
        NL_TE=1E12*NL_EB
        NL_TB=1E12*NL_EB
    else :
        sigma_amin=exp['noise_reduced']
        beam_amin=exp['beam_reduced']
        dir="noiseLensing"
        ell1,NL_TT=np.loadtxt('%s/lensNoise_TTTT_%d-%d_sigma%.02f_beam%.02f.txt'%(dir,lmin,lmax,sigma_amin,beam_amin),unpack=True)
        ell1,NL_EE=np.loadtxt('%s/lensNoise_EEEE_%d-%d_sigma%.02f_beam%.02f.txt'%(dir,lmin,lmax,sigma_amin,beam_amin),unpack=True)
        ell1,NL_EB=np.loadtxt('%s/lensNoise_EBEB_%d-%d_sigma%.02f_beam%.02f.txt'%(dir,lmin,lmax,sigma_amin,beam_amin),unpack=True)
        ell1,NL_TB=np.loadtxt('%s/lensNoise_TBTB_%d-%d_sigma%.02f_beam%.02f.txt'%(dir,lmin,lmax,sigma_amin,beam_amin),unpack=True)
        ell1,NL_TE=np.loadtxt('%s/lensNoise_TETE_%d-%d_sigma%.02f_beam%.02f.txt'%(dir,lmin,lmax,sigma_amin,beam_amin),unpack=True)

    clkappa=np.zeros_like(ell1)
    clkappa[:len(ell)]=cl_fid['kappa']
    m=np.log(cl_fid['kappa'][-1]/cl_fid['kappa'][-2])/np.log(ell[-1]/ell[-2])
    clkappa[len(ell):]=cl_fid['kappa'][-1]*np.exp(m*np.log(ell1[len(ell):]/ell[-1]))

    if DEBUG :
        plt.plot(ell1,clkappa,'k-')
        plt.plot(ell1,NL_TT,'r-')
        plt.plot(ell1,NL_EE,'g-')
        plt.plot(ell1,NL_EB,'b-')
        plt.plot(ell1,1./(1./NL_EE+1./NL_EB),'y-')
        plt.plot(ell1,1./(1./NL_EE+1./NL_EB+1./NL_TT),'c-')
        plt.ylim([1E-9,3E-6])
        plt.xlim([2,8E3])
        plt.loglog()
        plt.show()

    return (ell1,NL_TT,NL_EE,NL_EB,NL_TB,NL_TE,clkappa)

#Set noise powerspectrum for SZ
def set_sz_noisepower(prefix_camb,exp) :
    print "Computing SZ noise power spectrum" 
    ell,cl_fid=read_cmb_powerspec(prefix_camb)
    cl_fid['TT']=cl_fid['TT']; cl_fid['TE']=cl_fid['TE']; cl_fid['EE']=cl_fid['EE'];

    nu=np.array(exp['freq'])
    noise=np.array(exp['noise'])
    theta_beam=np.array(exp['beam'])
    n_nu=len(nu);

    noise*=1./(RAD2AMIN*TCMB*1E6)
    theta_beam*=1./(RAD2AMIN*2.355)
    theta2_beam=theta_beam[:,None]**2+theta_beam[None,:]**2
    beam_array=np.exp(0.5*(ell**2)[:,np.newaxis,np.newaxis]*theta2_beam)

    cl_noise_mat=np.zeros([len(cl_fid['TT']),n_nu,n_nu])
    #TT
    cl_noise_mat[:,:,:]=cl_fid['TT'][:,None,None]/beam_array
    for inu in np.arange(n_nu) :
        cl_noise_mat[:,inu,inu]+=noise[inu]**2
    cl_noise_mat*=beam_array

    if DEBUG :
        for inu in np.arange(n_nu) :
            plt.plot(ell,cl_noise_mat[:,inu,inu],'r-')
            plt.ylim([1E-19,1E-9])
            plt.loglog()
        plt.show()
    fnu_tsz=fnu_sz(nu,0)
    fnu_ksz=fnu_sz(nu,1)
    icl_tt=np.sum(fnu_tsz*np.linalg.solve(cl_noise_mat,
                                          (np.ones(len(ell)))[:,np.newaxis]*fnu_tsz),axis=1)
    icl_tk=np.sum(fnu_tsz*np.linalg.solve(cl_noise_mat,
                                          (np.ones(len(ell)))[:,np.newaxis]*fnu_ksz),axis=1)
    icl_kk=np.sum(fnu_ksz*np.linalg.solve(cl_noise_mat,
                                          (np.ones(len(ell)))[:,np.newaxis]*fnu_ksz),axis=1)

    return ell,np.array([[icl_tt,icl_tk],[icl_tk,icl_kk]]),np.array([fnu_tsz,fnu_ksz]),theta_beam,noise,n_nu

def get_covariance_matched_filter(exp,prefix_camb,fields_lensing) :
    prefix=exp['name']+'_save_'+fields_lensing

    if os.path.isfile(prefix+'_mf_covar.txt') :
        data=np.loadtxt(prefix+'_mf_covar.txt',unpack=True)
        lth_arr=data[0]
        cov_arr_c=np.zeros([3,3,len(lth_arr)])
        cov_arr_c[0,0,:]=data[1]
        cov_arr_c[0,1,:]=data[2]
        cov_arr_c[1,0,:]=data[2]
        cov_arr_c[1,1,:]=data[3]
        cov_arr_c[2,2,:]=data[4]
    else :
        lth_arr=LTH_MIN+(LTH_MAX-LTH_MIN)*(np.arange(NTH)+0.5)/NTH

        #Fourier-space profiles for tSZ and kSZ
        lk_arr,pk_arr,lx_arr,px_arr=compute_fourier_profiles()
        l_cmb,icl_matrix,fnu_vec,theta_beam,noise_sigma,n_nu=set_sz_noisepower(prefix_camb,exp)

        l_cmbk,nl_tt,nl_ee,nl_eb,nl_tb,nl_te,cl_kappa=set_kappa_noisepower(prefix_camb,exp,False)
        inl_kappa=np.zeros_like(nl_tt)
        if fields_lensing=='TT' :
            inl_kappa=1./nl_tt
        elif fields_lensing=='Pol' :
            inl_kappa=1./nl_ee+1./nl_eb
        elif fields_lensing=='All' :
            inl_kappa=1./nl_tt+1./nl_ee+1./nl_eb+1./nl_tb+1./nl_te
        inl_kappa=1./(cl_kappa+1./inl_kappa)
        tilt_noisekappa=np.log(inl_kappa[-1]/inl_kappa[-2])/(l_cmbk[-1]-l_cmbk[-2])

        cov_arr_c=intg.integ_clusternoise(LK_MIN,LK_MAX,lk_arr,pk_arr,fnu_vec,theta_beam,noise_sigma,
                                          int(l_cmb[-1]),icl_matrix,int(l_cmbk[-1]),tilt_noisekappa,
                                          inl_kappa,lth_arr);

        np.savetxt(prefix+'_mf_covar.txt',np.transpose([lth_arr,cov_arr_c[0,0],cov_arr_c[0,1],
                                                        cov_arr_c[1,1],cov_arr_c[2,2]]))
                                                        
    return lth_arr,cov_arr_c
