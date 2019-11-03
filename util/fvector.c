#include "basic.h"
#include "../nr/nr.h"

float *fvector(int nl,int nh)
{
	float *v;

	v=(float *)malloc((unsigned) (nh-nl+1)*sizeof(float));
	if (!v) nrerror("allocation failure in vector()");
	return v-nl;
}


void free_fvector(float *v,int nl,int nh)
{
	free((char*) (v+nl));
}

