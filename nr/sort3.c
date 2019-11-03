#include "basic.h"

void sort3(unsigned long n, double ra[], double rb[], double rc[])
{
	void indexx(unsigned long n, double arr[], unsigned long indx[]);
	unsigned long j,*iwksp;
	double *wksp;

	iwksp=lvector(1,n);
	wksp=vector(1,n);
	indexx(n,ra,iwksp);
	for (j=1;j<=n;j++) wksp[j]=ra[j];
	for (j=1;j<=n;j++) ra[j]=wksp[iwksp[j]];
	for (j=1;j<=n;j++) wksp[j]=rb[j];
	for (j=1;j<=n;j++) rb[j]=wksp[iwksp[j]];
	for (j=1;j<=n;j++) wksp[j]=rc[j];
	for (j=1;j<=n;j++) rc[j]=wksp[iwksp[j]];
	free_vector(wksp,1,n);
	free_lvector(iwksp,1,n);
}

/* sort3dii is based on sort3, but with rb and rc int rather than double */

void sort3dii(unsigned long n, double ra[], int rb[], int rc[])
{
	void indexx(unsigned long n, double arr[], unsigned long indx[]);
	unsigned long j,*iwksp;
        int *wksp_bc;
	double *wksp_a;

	iwksp=lvector(1,n);
	wksp_a=vector(1,n);
	wksp_bc=ivector(1,n);
	indexx(n,ra,iwksp);
	for (j=1;j<=n;j++) wksp_a[j]=ra[j];
	for (j=1;j<=n;j++) ra[j]=wksp_a[iwksp[j]];
	for (j=1;j<=n;j++) wksp_bc[j]=rb[j];
	for (j=1;j<=n;j++) rb[j]=wksp_bc[iwksp[j]];
	for (j=1;j<=n;j++) wksp_bc[j]=rc[j];
	for (j=1;j<=n;j++) rc[j]=wksp_bc[iwksp[j]];
	free_ivector(wksp_bc,1,n);
	free_vector(wksp_a,1,n);
	free_lvector(iwksp,1,n);
}
