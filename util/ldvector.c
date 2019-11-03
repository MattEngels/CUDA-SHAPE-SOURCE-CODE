#include "basic.h"
#include "../nr/nr.h"


long double *ldvector(int nl,int nh)
{
  long double *v;

  v=(long double *)malloc((unsigned) (nh-nl+1)*sizeof(long double));
  if (!v) nrerror("allocation failure in ucvector()");
  return v-nl;
}
