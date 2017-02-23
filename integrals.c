#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <numpy/arrayobject.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>

#define N_INTEG 1000
//#define S_LSCATTER 0.127
//High precision parameters
#define RELERR 1.0E-6
#define ABSERR 0.
#define PRECISION GSL_INTEG_GAUSS61
#define FAC_SIGMA 10.
/*
//Low precision parameters
#define RELERR 1.0E-4
#define ABSERR 1.0E-6
//#define PRECISION GSL_INTEG_GAUSS15
#define PRECISION GSL_INTEG_GAUSS61
#define FAC_SIGMA 10.
*/

static double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
  return (double *)arrayin->data;
}

static int  not_doublevector(PyArrayObject *vec)  {
  if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
    PyErr_SetString(PyExc_ValueError,
		    "In not_doublevector: array must be of type Float and 1 dimensional (n).");
    return 1;  }
  return 0;
}

typedef struct {
  double lny_mean;
  double sigma_lny;
  double sigma_N;
  double q;
} param_integ;

static double integrand_chi(double lny,void *param)
{
  param_integ *par=(param_integ *)param;
  double y=exp(lny);
  //  double x=(lny-par->lny_mean)/S_LSCATTER;
  //  double p1=exp(-0.5*x*x)/(sqrt(2*M_PI)*S_LSCATTER);
  double x=(lny-par->lny_mean)/par->sigma_lny;
  double p1=exp(-0.5*x*x)/(sqrt(2*M_PI)*par->sigma_lny);
  double p2=0.5*(1+erf((y-par->q*par->sigma_N)/(M_SQRT2*par->sigma_N)));

  return p1*p2;
}

static double integrate_chi(double y_signal,double sigma_N,double q,double sigma_lny)
{
  double result,eresult;
  param_integ par;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(N_INTEG);
  par.lny_mean=log(y_signal);
  par.sigma_lny=sigma_lny;
  par.sigma_N=sigma_N;
  par.q=q;
  F.function=&integrand_chi;
  F.params=&par;

  gsl_integration_qag(&F,par.lny_mean-FAC_SIGMA*sigma_lny,par.lny_mean+FAC_SIGMA*sigma_lny,
		      ABSERR,RELERR,N_INTEG,PRECISION,w,&result,&eresult);

  gsl_integration_workspace_free(w);

  return result;
}

static PyObject *integrals_integ_chi(PyObject *self,PyObject *args)
{
  double y_signal,sigma_N,q,sigma_lny,result;

  if (!PyArg_ParseTuple(args,"dddd",&y_signal,&sigma_N,&q,&sigma_lny))
    return NULL;

  result=integrate_chi(y_signal,sigma_N,q,sigma_lny);

  return Py_BuildValue("d",result);
}

static double integrand_pdf(double lny,void *param)
{
  param_integ *par=(param_integ *)param;
  double y=exp(lny);
  double x1=(lny-par->lny_mean)/par->sigma_lny;
  double x2=par->q-y/par->sigma_N;
  //  double p1=exp(-0.5*x1*x1)/(sqrt(2*M_PI)*par->sigma_lny);
  //  double p2=par->q*exp(-0.5*x2*x2)/sqrt(2*M_PI);
  //  return p1*p2;

  return par->q*exp(-0.5*(x1*x1+x2*x2))/(2*M_PI*par->sigma_lny);
}

static double integrate_pdf(double y_signal,double sigma_N,double q,double sigma_lny)
{
  double result,eresult;
  param_integ par;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(N_INTEG);
  par.lny_mean=log(y_signal);
  par.sigma_lny=sigma_lny;
  par.sigma_N=sigma_N;
  par.q=q;
  F.function=&integrand_pdf;
  F.params=&par;

  gsl_integration_qag(&F,par.lny_mean-FAC_SIGMA*sigma_lny,par.lny_mean+FAC_SIGMA*sigma_lny,
		      ABSERR,RELERR,N_INTEG,PRECISION,w,&result,&eresult);
  gsl_integration_workspace_free(w);

  return result;
}

static PyObject *integrals_integ_pdf(PyObject *self,PyObject *args)
{
  double y_signal,sigma_N,sigma_lny,q,result;

  if (!PyArg_ParseTuple(args,"dddd",&y_signal,&sigma_N,&q,&sigma_lny))
    return NULL;

  result=integrate_pdf(y_signal,sigma_N,q,sigma_lny);

  return Py_BuildValue("d",result);
}

typedef struct {
  int nk;
  double logk_min;
  double logk_max;
  double *lk;
  double **pk;
  gsl_spline **spline_pk;
  gsl_interp_accel **intacc_pk;
  int nnu;
  double **fnu;
  double *th_beam;
  double *noise_sigma;
  int lmax_sz;
  double ***icl_matrix_sz;
  int lmax_kappa;
  double *inl_kappa;
  double tilt_kappa;
} ParNoise;

static double sp_eval_safe(double lk,int nk,double *lkarr,double *pkarr,
			   gsl_spline *spline,gsl_interp_accel *intacc)
{
  if(lk>=lkarr[nk-1]) {
    return pkarr[nk-1]*pow(10.,log10(pkarr[nk-1]/pkarr[nk-2])*(lk-lkarr[nk-1])/(lkarr[nk-1]-lkarr[nk-2]));
  }
  else if(lk<=lkarr[0]) {
    return pkarr[0];
  }
  else {
    return gsl_spline_eval(spline,lk,intacc);
  }  
}

typedef struct {
  int ic1;
  int ic2;
  double theta;
  ParNoise *p;
} ParNoiseInteg;

static double integrand_sz(double lk,void *params)
{
  double icl;
  ParNoiseInteg *pi=(ParNoiseInteg *)params;
  ParNoise *p=pi->p;
  double k=pow(10.,lk);
  double l_k=k/pi->theta;
  double p1=k*sp_eval_safe(lk,p->nk,p->lk,p->pk[pi->ic1],p->spline_pk[pi->ic1],p->intacc_pk[pi->ic1]);
  double p2=k*sp_eval_safe(lk,p->nk,p->lk,p->pk[pi->ic2],p->spline_pk[pi->ic2],p->intacc_pk[pi->ic2]);

  if(l_k>p->lmax_sz) {
    int inu;
    icl=0;
    for(inu=0;inu<p->nnu;inu++) {
      double inoise=p->noise_sigma[inu]*exp(0.5*l_k*l_k*p->th_beam[inu]*p->th_beam[inu]);
      icl+=p->fnu[pi->ic1][inu]*p->fnu[pi->ic2][inu]/(inoise*inoise);
    }
  }
  else if(l_k<2)
    icl=p->icl_matrix_sz[pi->ic1][pi->ic2][0];
  else
    icl=p->icl_matrix_sz[pi->ic1][pi->ic2][(int)(l_k)-2];

  return p1*p2*icl;
}

static double integrand_kappa(double lk,void *params)
{
  double inl;
  ParNoiseInteg *pi=(ParNoiseInteg *)params;
  ParNoise *p=pi->p;
  double k=pow(10.,lk);
  double l_k=k/pi->theta;
  double p1=k*sp_eval_safe(lk,p->nk,p->lk,p->pk[2],p->spline_pk[2],p->intacc_pk[2]);
  
  if(l_k>p->lmax_kappa)
    inl=p->inl_kappa[p->lmax_kappa-2]*exp(p->tilt_kappa*(l_k-p->lmax_kappa));
  else if(l_k<2)
    inl=p->inl_kappa[0];
  else
    inl=p->inl_kappa[(int)(l_k)-2];

  return p1*p1*inl;
}

static double integrate_noise(ParNoise *p,double theta,int ic1,int ic2)
{
  double result,eresult;
  if(ic1==2) {
    if(ic2==2) { //kk
      ParNoiseInteg par;
      gsl_function F;
      gsl_integration_workspace *w=gsl_integration_workspace_alloc(N_INTEG);
      par.p=p;
      par.ic1=ic1;
      par.ic2=ic2;
      par.theta=theta;
      F.function=&integrand_kappa;
      F.params=&par;

      gsl_integration_qag(&F,p->logk_min,p->logk_max,
			  ABSERR,RELERR,N_INTEG,PRECISION,w,&result,&eresult);

      gsl_integration_workspace_free(w);
    }
    else //k-sz
      result=0;
  }
  else {
    if(ic2==2) //k-sz
      result=0;
    else { //sz-sz
      ParNoiseInteg par;
      gsl_function F;
      gsl_integration_workspace *w=gsl_integration_workspace_alloc(N_INTEG);
      par.p=p;
      par.ic1=ic1;
      par.ic2=ic2;
      par.theta=theta;
      F.function=&integrand_sz;
      F.params=&par;

      gsl_integration_qag(&F,p->logk_min,p->logk_max,
			  ABSERR,RELERR,N_INTEG,PRECISION,w,&result,&eresult);

      gsl_integration_workspace_free(w);
    }
  }

  return M_LN10*result/(8*M_PI*theta*theta);
}

static PyObject *integrals_integ_clusternoise(PyObject *self,PyObject *args)
{
  ParNoise *p=malloc(sizeof(ParNoise));
  PyArrayObject *olkarr,*opkarr,*ofnu,*oth_beam,*onoise_sigma,*oicl_matrix_sz,*oinl_kappa,*olth_arr;
  PyArrayObject *outa=NULL;
  double logk_min,logk_max,tilt_kappa;
  int ith,ic,ik,inu,il,lmax_sz,lmax_kappa;
  int dims[3];

  if(!PyArg_ParseTuple(args,"ddO!O!O!O!O!iO!idO!O!",
		       &logk_min,&logk_max,
		       &PyArray_Type,&olkarr,
		       &PyArray_Type,&opkarr,
		       &PyArray_Type,&ofnu,
		       &PyArray_Type,&oth_beam,
		       &PyArray_Type,&onoise_sigma,
		       &lmax_sz,
		       &PyArray_Type,&oicl_matrix_sz,
		       &lmax_kappa,&tilt_kappa,
		       &PyArray_Type,&oinl_kappa,
		       &PyArray_Type,&olth_arr))
    return NULL;

  p->logk_min=logk_min;
  p->logk_max=logk_max;
  p->nk=olkarr->dimensions[0];
  p->lk=malloc(p->nk*sizeof(double));
  for(ik=0;ik<p->nk;ik++)
    p->lk[ik]=*((double *)(olkarr->data+ik*olkarr->strides[0]));
  p->pk=malloc(3*sizeof(double *));
  p->spline_pk=malloc(3*sizeof(gsl_spline *));
  p->intacc_pk=malloc(3*sizeof(gsl_interp_accel *));
  for(ic=0;ic<3;ic++) {
    p->pk[ic]=malloc(p->nk*sizeof(double));
    for(ik=0;ik<p->nk;ik++)
      p->pk[ic][ik]=*((double *)(opkarr->data+ic*opkarr->strides[0]+ik*opkarr->strides[1]));
    p->spline_pk[ic]=gsl_spline_alloc(gsl_interp_cspline,p->nk);
    gsl_spline_init(p->spline_pk[ic],p->lk,p->pk[ic],p->nk);
    p->intacc_pk[ic]=gsl_interp_accel_alloc();
  }

  p->nnu=oth_beam->dimensions[0];
  p->fnu=malloc(2*sizeof(double *));
  for(ic=0;ic<2;ic++) {
    p->fnu[ic]=malloc(p->nnu*sizeof(double));
    for(inu=0;inu<p->nnu;inu++)
      p->fnu[ic][inu]=*((double *)(ofnu->data+ic*ofnu->strides[0]+inu*ofnu->strides[1]));
  }
  p->th_beam=malloc(p->nnu*sizeof(double));
  for(inu=0;inu<p->nnu;inu++)
    p->th_beam[inu]=*((double *)(oth_beam->data+inu*oth_beam->strides[0]));
  p->noise_sigma=malloc(p->nnu*sizeof(double));
  for(inu=0;inu<p->nnu;inu++)
    p->noise_sigma[inu]=*((double *)(onoise_sigma->data+inu*onoise_sigma->strides[0]));
  p->lmax_sz=lmax_sz;
  p->lmax_kappa=lmax_kappa;
  p->icl_matrix_sz=malloc(2*sizeof(double **));
  for(ic=0;ic<2;ic++) {
    int ic2;
    p->icl_matrix_sz[ic]=malloc(2*sizeof(double *));
    for(ic2=0;ic2<2;ic2++) {
      p->icl_matrix_sz[ic][ic2]=malloc(oicl_matrix_sz->dimensions[2]*sizeof(double));
      for(il=0;il<oicl_matrix_sz->dimensions[2];il++) {
	p->icl_matrix_sz[ic][ic2][il]=*((double *)(oicl_matrix_sz->data+ic*oicl_matrix_sz->strides[0]+
						   ic2*oicl_matrix_sz->strides[1]+il*oicl_matrix_sz->strides[2]));
      }
    }
  }
  p->inl_kappa=malloc(oinl_kappa->dimensions[0]*sizeof(double));
  for(il=0;il<oinl_kappa->dimensions[0];il++)
    p->inl_kappa[il]=*((double *)(oinl_kappa->data+il*oinl_kappa->strides[0]));
  p->tilt_kappa=tilt_kappa;

  dims[0]=3;
  dims[1]=3;
  dims[2]=olth_arr->dimensions[0];
  outa=(PyArrayObject *)PyArray_FromDims(3,dims,PyArray_DOUBLE);

  for(ith=0;ith<olth_arr->dimensions[0];ith++) {
    double th=pow(10.,*((double *)(olth_arr->data+ith*olth_arr->strides[0])))*M_PI/(180*60);
    double ic00=integrate_noise(p,th,0,0);
    double ic01=integrate_noise(p,th,0,1);
    double ic11=integrate_noise(p,th,1,1);
    double ic22=integrate_noise(p,th,2,2);
    double idet=1./(ic00*ic11-ic01*ic01);
    double c00= ic11*idet;
    double c01=-ic01*idet;
    double c11= ic00*idet;
    double c22=1./ic22;
    *((double *)(outa->data+0*outa->strides[0]+0*outa->strides[1]+ith*outa->strides[2]))=c00;
    *((double *)(outa->data+0*outa->strides[0]+1*outa->strides[1]+ith*outa->strides[2]))=c01;
    *((double *)(outa->data+0*outa->strides[0]+2*outa->strides[1]+ith*outa->strides[2]))=0.;
    *((double *)(outa->data+1*outa->strides[0]+0*outa->strides[1]+ith*outa->strides[2]))=c01;
    *((double *)(outa->data+1*outa->strides[0]+1*outa->strides[1]+ith*outa->strides[2]))=c11;
    *((double *)(outa->data+1*outa->strides[0]+2*outa->strides[1]+ith*outa->strides[2]))=0.;
    *((double *)(outa->data+2*outa->strides[0]+0*outa->strides[1]+ith*outa->strides[2]))=0.;
    *((double *)(outa->data+2*outa->strides[0]+1*outa->strides[1]+ith*outa->strides[2]))=0.;
    *((double *)(outa->data+2*outa->strides[0]+2*outa->strides[1]+ith*outa->strides[2]))=c22;
  }

  for(ic=0;ic<3;ic++) {
    gsl_spline_free(p->spline_pk[ic]);
    gsl_interp_accel_free(p->intacc_pk[ic]);
    free(p->pk[ic]);
  }
  free(p->lk);
  free(p->pk);
  free(p->spline_pk);
  free(p->intacc_pk);

  for(ic=0;ic<2;ic++) {
    int ic2;
    for(ic2=0;ic2<2;ic2++) {
      free(p->icl_matrix_sz[ic][ic2]);
    }
    free(p->icl_matrix_sz[ic]);
    free(p->fnu[ic]);
  }
  free(p->icl_matrix_sz);
  free(p->fnu);
  free(p->th_beam);
  free(p->noise_sigma);
  free(p->inl_kappa);
  free(p);

  return PyArray_Return(outa);
}

static PyObject *integrals_integ_pdf_array(PyObject *self,PyObject *args)
{
  double sigma_lny=0.127;
  double *q;
  int dims[3];
  PyArrayObject *qa,*ysign,*ysigm,*outa=NULL;

  if (!PyArg_ParseTuple(args,"O!O!O!d",
			&PyArray_Type,&ysign,
			&PyArray_Type,&ysigm,
			&PyArray_Type,&qa,&sigma_lny))
    return NULL;

  if(NULL==qa) return NULL;
  if(NULL==ysign) return NULL;
  if(NULL==ysigm) return NULL;

  if(not_doublevector(qa)) return NULL;

  dims[0]=qa->dimensions[0];
  dims[1]=ysign->dimensions[0];
  dims[2]=ysign->dimensions[1];

  q=pyvector_to_Carrayptrs(qa);

  outa=(PyArrayObject *)PyArray_FromDims(3,dims,PyArray_DOUBLE);
#pragma omp parallel default(none)		\
  shared(dims,q,ysign,ysigm,outa,sigma_lny)
  {
    int iq;

#pragma omp for
    for(iq=0;iq<dims[0];iq++) {
      int iz;
      for(iz=0;iz<dims[1];iz++) {
	int ilm;
	double *ysgn=(double *)(ysign->data+iz*ysign->strides[0]);
	double *ysgm=(double *)(ysigm->data+iz*ysigm->strides[0]);
	for(ilm=0;ilm<dims[2];ilm++) {
	  double *pdfel=(double *)(outa->data+
				   iq*outa->strides[0]+
				   iz*outa->strides[1]+
				   ilm*outa->strides[2]);
	  *pdfel=integrate_pdf(ysgn[ilm],ysgm[ilm],q[iq],sigma_lny);
	}
      }
    }
  }

  return PyArray_Return(outa);
}

static PyMethodDef IntegralsMethods[] = {
  {"integ_chi",integrals_integ_chi,METH_VARARGS,"Integrate chi"},
  {"integ_pdf",integrals_integ_pdf,METH_VARARGS,"Integrate pdf"},
  {"integ_pdf_array",integrals_integ_pdf_array,METH_VARARGS,"Integrate pdf array"},
  {"integ_clusternoise",integrals_integ_clusternoise,METH_VARARGS,"Integrate pdf array"},
  {NULL,NULL,0,NULL}
};

PyMODINIT_FUNC initintegrals(void)
{
  (void)Py_InitModule("integrals",IntegralsMethods);
  import_array();
}
