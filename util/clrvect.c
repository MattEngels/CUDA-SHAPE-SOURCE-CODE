#include "basic.h"

void clrvect( double *v, int i1, int i2)
{
  int i;

  for (i=i1;i<=i2;i++)
	v[i] = 0.0;
}
