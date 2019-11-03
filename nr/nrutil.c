#include "basic.h"
#define NR_END 1

void nrerror(const char error_text[])
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

double *vector(int nl,int nh)
{
	double *v;

	v=(double *) malloc((size_t) (nh-nl+1)*sizeof(double));
	if (!v) nrerror("allocation failure in vector()");
	return v-nl;
}

double *vector_zero(int nl,int nh)
{
	double *v;

	v=(double *) malloc((size_t) (nh-nl+1)*sizeof(double));
	if (!v) nrerror("allocation failure in vector()");
	for (int i=nl; i<nh; i++)
		v[i]=0.0;
	return v-nl;
}

int *ivector(int nl,int nh)
{
	int *v;

	v=(int *)malloc((unsigned) (nh-nl+1)*sizeof(int));
	if (!v) nrerror("allocation failure in ivector()");
	return v-nl;
}

unsigned long *lvector(int nl, int nh)
{
	unsigned long *v;

	v=(unsigned long *)malloc((size_t) ((nh-nl+1)*sizeof(long)));
	if (!v) nrerror("allocation failure in lvector()");
	return v-nl;
}

double *dvector(int nl,int nh)
{
	double *v;

	v=(double *) malloc((unsigned) (nh-nl+1)*sizeof(double));
	if (!v) nrerror("allocation failure in dvector()");
	return v-nl;
}

double **matrix(int nrl,int nrh,int ncl,int nch)
{
	int i;
	double **m;

	m=(double **) malloc((unsigned) (nrh-nrl+1)*sizeof(double*));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(double *) malloc((unsigned) (nch-ncl+1)*sizeof(double));
		if (!m[i]) nrerror("allocation failure 2 in matrix()");
		m[i] -= ncl;
	}
	return m;
}

double **matrix_zero(int nrl,int nrh,int ncl,int nch)
{
	int i;
	double **m;

	m=(double **) malloc((unsigned) (nrh-nrl+1)*sizeof(double*));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(double *) malloc((unsigned) (nch-ncl+1)*sizeof(double));
		if (!m[i]) nrerror("allocation failure 2 in matrix()");
		m[i] -= ncl;
	}

	for (i=nrl; i<nrh; i++)
		for (int j=ncl; j<nch; j++)
			m[i][j]=0.0;
	return m;
}


double **dmatrix(int nrl,int nrh,int ncl,int nch)
{
	int i;
	double **m;

	m=(double **) malloc((unsigned) (nrh-nrl+1)*sizeof(double*));
	if (!m) nrerror("allocation failure 1 in dmatrix()");
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(double *) malloc((unsigned) (nch-ncl+1)*sizeof(double));
		if (!m[i]) nrerror("allocation failure 2 in dmatrix()");
		m[i] -= ncl;
	}
	return m;
}

int **imatrix(int nrl,int nrh,int ncl,int nch)
{
	int i,**m;

	m=(int **)malloc((unsigned) (nrh-nrl+1)*sizeof(int*));
	if (!m) nrerror("allocation failure 1 in imatrix()");
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(int *)malloc((unsigned) (nch-ncl+1)*sizeof(int));
		if (!m[i]) nrerror("allocation failure 2 in imatrix()");
		m[i] -= ncl;
	}
	return m;
}

double **submatrix(double **a,int oldrl,int oldrh,int oldcl,int oldch,int
		newrl,int newcl)
{
	int i,j;
	double **m;

	m=(double **) malloc((unsigned) (oldrh-oldrl+1)*sizeof(double*));
	if (!m) nrerror("allocation failure in submatrix()");
	m -= newrl;

	for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+oldcl-newcl;

	return m;
}

double **convert_matrix(double *a,int nrl,int nrh,int ncl,int nch)
{
	int i,j,nrow,ncol;
	double **m;

	nrow=nrh-nrl+1;
	ncol=nch-ncl+1;
	m = (double **) malloc((unsigned) (nrow)*sizeof(double*));
	if (!m) nrerror("allocation failure in convert_matrix()");
	m -= nrl;
	for(i=0,j=nrl;i<=nrow-1;i++,j++) m[j]=a+ncol*i-ncl;
	return m;
}

double ***d3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh)
/* allocate a double 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
	int i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
	double ***t;

	/* allocate pointers to pointers to rows */
	t=(double ***) malloc((size_t)((nrow+NR_END)*sizeof(double**)));
	if (!t) nrerror("allocation failure 1 in d3tensor()");
	t += NR_END;
	t -= nrl;

	/* allocate pointers to rows and set pointers to them */
	t[nrl]=(double **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double*)));
	if (!t[nrl]) nrerror("allocation failure 2 in d3tensor()");
	t[nrl] += NR_END;
	t[nrl] -= ncl;

	/* allocate rows and set pointers to them */
	t[nrl][ncl]=(double *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(double)));
	if (!t[nrl][ncl]) nrerror("allocation failure 3 in d3tensor()");
	t[nrl][ncl] += NR_END;
	t[nrl][ncl] -= ndl;

	for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
	for(i=nrl+1;i<=nrh;i++) {
		t[i]=t[i-1]+ncol;
		t[i][ncl]=t[i-1][ncl]+ncol*ndep;
		for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
	}

	/* return pointer to array of pointers to rows */
	return t;
}

void free_vector(double *v,int nl,int nh)
{
	free((char*) (v+nl));
}

void free_ivector(int *v,int nl,int nh)
{
	free((char*) (v+nl));
}

void free_lvector(unsigned long *v,int nl,int nh)
{
	free((char*) (v+nl));
}

void free_dvector(double *v,int nl,int nh)
{
	free((char*) (v+nl));
}

void free_matrix(double **m,int nrl,int nrh,int ncl,int nch)
{
	int i;

	for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
	free((char*) (m+nrl));
}

void free_dmatrix(double **m,int nrl,int nrh,int ncl,int nch)
{
	int i;

	for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
	free((char*) (m+nrl));
}

void free_imatrix(int **m,int nrl,int nrh,int ncl,int nch)
{
	int i;

	for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
	free((char*) (m+nrl));
}

void free_submatrix(double **b,int nrl,int nrh,int ncl,int nch)
{
	free((char*) (b+nrl));
}

void free_convert_matrix(double **b,int nrl,int nrh,int ncl,int nch)
{
	free((char*) (b+nrl));
}

void free_d3tensor(double ***t, int nrl, int nrh, int ncl, int nch,
		/* free a double d3tensor allocated by d3tensor() */
		int ndl, int ndh)
{
	free((char*) (t[nrl][ncl]+ndl-NR_END));
	free((char*) (t[nrl]+ncl-NR_END));
	free((char*) (t+nrl-NR_END));
}
