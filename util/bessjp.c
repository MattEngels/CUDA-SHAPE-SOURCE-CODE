/***************************************************************************

								 bessjp.c

Returns the derivative of J_n(x).

***************************************************************************/

#include "basic.h"
#include "../nr/nr.h"
#include "util.h"

double bessjp( int n, double x)
{
  if (n==0)
	return -bessj(1,x);
  else
	return 0.5*(bessj(n-1,x)-bessj(n+1,x));
}
