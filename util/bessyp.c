/***************************************************************************

								 bessjp.c

Returns the derivative of Y_n(x).

***************************************************************************/

#include "basic.h"
#include "../nr/nr.h"

double bessyp( int n, double x)
{
  if (n==0)
	return -bessy(1,x);
  else
	return 0.5*(bessy(n-1,x)-bessy(n+1,x));
}
