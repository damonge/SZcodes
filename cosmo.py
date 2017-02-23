import numpy as np
import py_cosmo_mad as csm
from parameters import *
import os

ST_A=0.322
ST_a=0.707
ST_p=0.3
JP_a=1.529
JP_b=0.704
JP_c=0.412
JEN_B=0.315
JEN_b=0.61
JEN_q=3.8
WAR_A=0.7234
WAR_a=1.625
WAR_b=0.2538
WAR_c=1.1982
TINKER_A_500=0.215
TINKER_Aexp_500=-0.14
TINKER_a_500=1.585
TINKER_aexp_500=-0.06
TINKER_b_500=1.960
TINKER_bexp_500=-0.128
TINKER_c_500=1.385
TINKER_A_200=0.186
TINKER_Aexp_200=-0.14
TINKER_a_200=1.47
TINKER_aexp_200=-0.06
TINKER_b_200=2.57
TINKER_bexp_200=-0.0107
TINKER_c_200=1.19
WATSON_A=0.282
WATSON_ALPHA=2.163
WATSON_BETA=1.406
WATSON_GAMMA=1.210

def g_nu(sigma,z,typ) :
    if typ=="PS" :
        nu=DELTA_C/sigma
        return np.sqrt(2/np.pi)*nu*np.exp(-nu**2/2)
    elif typ=="ST" :
        nu=DELTA_C/sigma
        return ST_A*np.sqrt(2*ST_a/np.pi)*(1+(1./(ST_a*nu**2))**ST_p)*nu*np.exp(-ST_a*nu**2/2)
    elif typ=="JP" :
        nu=DELTA_C/sigma
        return (JP_a*JP_b*nu**JP_b+2*JP_c*nu**2*(1+JP_a*nu**JP_b))*np.exp(-JP_c*nu**2)/(1+JP_a*nu**JP_b)**2
    elif typ=="Jenkins" :
        return JEN_B*np.exp(-(np.fabs(np.log(1./sigma)+JEN_b))**JEN_q)
    elif typ=="Warren" :
        return WAR_A*(1./sigma**WAR_a+WAR_b)*np.exp(-WAR_c/sigma**2)
    elif typ=="Tinker200" :
        A=TINKER_A_200*(1+z)**TINKER_Aexp_200
        a=TINKER_a_200*(1+z)**TINKER_aexp_200
        b=TINKER_b_200*(1+z)**TINKER_bexp_200
        c=TINKER_c_200
        return A*(1+(b/sigma)**a)*np.exp(-c/sigma**2)
    elif typ=="Tinker500" :
        A=TINKER_A_500*(1+z)**TINKER_Aexp_500
        a=TINKER_a_500*(1+z)**TINKER_aexp_500
        b=TINKER_b_500*(1+z)**TINKER_bexp_500
        c=TINKER_c_500
        return A*(1+(b/sigma)**a)*np.exp(-c/sigma**2)
    elif typ=="Watson" :
        return WATSON_A*(1+(WATSON_BETA/sigma)**WATSON_ALPHA)*np.exp(-WATSON_GAMMA/sigma**2)

PREFIX_SAVE="test_sav"

def write_class_param_file(cospar,z_arr) :
    """ Generates CLASS param file """
    hh=cospar['hh']
    och2=cospar['och2']
    obh2=cospar['obh2']
    w0=cospar['w0']
    wa=cospar['wa']
    ns=cospar['ns']
    a_s=cospar['a_s']
    tau=cospar['tau']
    mnu=cospar['mnu']

    strout="#CLASS param file by GoFish\n"
    strout+="h = %lE\n"%hh
    strout+="T_cmb = 2.725\n"
    strout+="omega_b = %lE\n"%obh2
    strout+="omega_cdm = %lE\n"%och2
    if mnu>0 :
        strout+="N_ur = 2.0328\n"
        strout+="N_ncdm = 1\n"
        strout+="m_ncdm = %lE \n"%(mnu*0.001)
    else :
        strout+="N_ur = 3.046\n"
    if (w0!=-1.) or (wa!=0.0):
        strout+="Omega_fld = %lE\n"%(1-(och2+obh2)/hh**2)
        if wa<0 :
            strout+="w0_fld = %lE\n"%(w0-0.00001)
        elif wa>0 :
            strout+="w0_fld = %lE\n"%(w0+0.00001)
        else :
            strout+="w0_fld = %lE\n"%w0
        strout+="wa_fld = %lE\n"%wa
        strout+="cs2_fld = 1\n"
    strout+="Omega_k = 0.\n"

    strout+="YHe = BBN\n"
    strout+="recombination = RECFAST\n"
    strout+="reio_parametrization = reio_camb\n"
    strout+="tau_reio = %lE\n"%tau
    strout+="output = mPk\n"
    strout+="ic = ad\n"
    strout+="gauge = synchronous\n"
    strout+="P_k_ini type = analytic_Pk\n"
    strout+="k_pivot = 0.05\n"
    strout+="A_s = %lE\n"%(a_s*1E-9)
    strout+="n_s = %lE\n"%ns
    strout+="alpha_s = 0.\n"
    strout+="l_max_scalars = 5000\n"
    strout+="l_max_tensors = 5000\n"
    strout+="l_max_lss = 500\n"
    strout+="P_k_max_h/Mpc = 10.\n"
#    strout+="z_pk = 0.0\n"
    strout+="z_pk =  %lf"%(z_arr[0])
    for z in z_arr[1:] :
        strout+=",%lf "%z
    strout+="\n"
    strout+="root = "+PREFIX_SAVE+"\n"
    strout+="output_binary = 0\n"
    strout+="format = class\n"
    strout+="write background = no\n"
    strout+="write thermodynamics = no\n"
    strout+="write primordial = no\n"
    strout+="write parameters = yeap\n"
    strout+="headers = no\n"
    strout+="l_switch_limber= 10.\n"   #%(par.lmin_limber)
    strout+="l_switch_limber_for_cl_density= 1000\n"
    strout+="l_switch_limber_for_cl_lensing= 1000\n"
    strout+="selection_sampling_bessel=2.\n"
    strout+="k_step_trans_scalars=0.4\n"
    strout+="q_linstep=0.4\n"
    strout+="k_scalar_max_tau0_over_l_max= 2.\n"
    strout+="selection_tophat_edge= 0.01\n"
    strout+="z_min_k= 0.1\n"
    strout+="perturb_sampling_stepsize=0.06\n"
    strout+="write warnings = y\n"
    strout+="input_verbose = 1\n"
    strout+="background_verbose = 1\n"
    strout+="thermodynamics_verbose = 1\n"
    strout+="perturbations_verbose = 1\n"
    strout+="transfer_verbose = 1\n"
    strout+="primordial_verbose = 1\n"
    strout+="spectra_verbose = 1\n"
    strout+="nonlinear_verbose = 1\n"
    strout+="lensing_verbose = 1\n"
    strout+="output_verbose = 1\n"
    f=open(PREFIX_SAVE+"_param.ini","w")
    f.write(strout)
    f.close()
    os.system("./class_mod "+PREFIX_SAVE+"_param.ini >> log_class")

    return strout

def get_mfunc(cospar,z_min,z_max,nz,lm_min,lm_max,nm,fsky) :
    z_arr=z_min+(z_max-z_min)*(np.arange(nz)+0.5)/nz
    z0_arr=z_min+(z_max-z_min)*(np.arange(nz)+0.0)/nz
    zf_arr=z_min+(z_max-z_min)*(np.arange(nz)+1.0)/nz

    pcs=csm.PcsPar()
    if DEBUG :
        pcs.set_verbosity(1)
    else :
        pcs.set_verbosity(0)
    om=(cospar['och2']+cospar['obh2']+cospar['mnu']*0.001/93.14)/cospar['hh']**2
    ob=cospar['obh2']/cospar['hh']**2
    pcs.background_set(om,1-om,ob,cospar['w0'],cospar['wa'],cospar['hh'],TCMB)
    write_class_param_file(cospar,z_arr)

    m_arr  =10.**(lm_min+(lm_max-lm_min)*(np.arange(nm)+0.5)/nm)
    r_arr=np.array([pcs.M2R(m) for m in m_arr])
    m2_arr=10.**(lm_min+(lm_max-lm_min)*(np.arange(nm+1)+0.0)/nm)
    lmarr=np.log10(m2_arr)
    r2_arr=np.array([pcs.M2R(m) for m in m2_arr])

    nm_arr=np.zeros([nz,nm])
    for iz in np.arange(nz) :
        z=z_arr[iz]; z0=z0_arr[iz]; zf=zf_arr[iz]
        r0=pcs.radial_comoving_distance(1./(1+z0));
        rf=pcs.radial_comoving_distance(1./(1+zf));
        pcs.set_linear_pk(PREFIX_SAVE+"z%d_pk.dat"%(iz+1),-4.,3.,0.01,cospar['ns'],-1)
        volz=4*np.pi*fsky*(rf**3-r0**3)/(3*(zf-z0)) #pcs.radial_comoving_distance(1./(1+z))**2/pcs.hubble(1./(1+z))
        sigM_arr=np.sqrt(np.array([pcs.sig0_L(r,r,'TopHat','TopHat') for r in  r_arr]))
        sig2_arr=np.sqrt(np.array([pcs.sig0_L(r,r,'TopHat','TopHat') for r in r2_arr]))
        dlsdlm_arr=-np.log10(sig2_arr[1:]/sig2_arr[:-1])/(lmarr[1:]-lmarr[:-1])
        nm_arr[iz,:]=(np.log(10.)*volz*dlsdlm_arr*om*RHOCRIT0*
                      g_nu(sigM_arr,z,cospar['mfName'])/m_arr)
    os.system("rm "+PREFIX_SAVE+"*")

    return pcs,nm_arr

def get_s8(cp) :
    cospar=cp.copy()
    pcs=csm.PcsPar()
    if DEBUG :
        pcs.set_verbosity(1)
    else :
        pcs.set_verbosity(0)
    om=(cospar['och2']+cospar['obh2']+cospar['mnu']*0.001/93.14)/cospar['hh']**2
    ob=cospar['obh2']/cospar['hh']**2
    pcs.background_set(om,1-om,ob,cospar['w0'],cospar['wa'],cospar['hh'],TCMB)
    write_class_param_file(cospar,np.array([0]))
    pcs.set_linear_pk(PREFIX_SAVE+"pk.dat",-4.,3.,0.01,cospar['ns'],-1)
    
    return np.sqrt(pcs.sig0_L(8,8,'TopHat','TopHat'))
