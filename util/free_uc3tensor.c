#include "basic.h"
#include "../nr/nr.h"

#define NR_END 1
#define FREE_ARG char*

void free_uc3tensor(unsigned char ***t, int nrl, int nrh, int ncl, int nch,
	int ndl, int ndh)
/* free a unsigned char uc3tensor allocated by uc3tensor() */
{
	free((FREE_ARG) (t[nrl][ncl]+ndl-NR_END));
	free((FREE_ARG) (t[nrl]+ncl-NR_END));
	free((FREE_ARG) (t+nrl-NR_END));
}
