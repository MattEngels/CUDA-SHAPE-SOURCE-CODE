#include "basic.h"


void free_ucvector(unsigned char *v,int nl,int nh)
{
	free((char*) (v+nl));
}
