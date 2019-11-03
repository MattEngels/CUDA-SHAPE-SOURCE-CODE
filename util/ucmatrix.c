#include "basic.h"
#include "../nr/nr.h"

unsigned char **ucmatrix(int nrl,int nrh,int ncl,int nch)
{
	int i;
	unsigned char **m;

	m=(unsigned char **) malloc((unsigned) (nrh-nrl+1)*sizeof(unsigned char*)); 
	if (!m) nrerror("allocation failure 1 in ucmatrix()");
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(unsigned char *) malloc((unsigned) (nch-ncl+1)*sizeof(unsigned char)); 
		if (!m[i]) nrerror("allocation failure 2 in ucmatrix()");
		m[i] -= ncl;
	}
	return m;
}
