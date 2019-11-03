#include "basic.h"
#include "../nr/nr.h"


unsigned char *ucvector(int nl,int nh)
{
	unsigned char *v;

	v=(unsigned char *)malloc((unsigned) (nh-nl+1)*sizeof(unsigned char));
	if (!v) nrerror("allocation failure in ucvector()");
	return v-nl;
}
