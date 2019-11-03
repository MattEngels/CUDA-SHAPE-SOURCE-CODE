/***************************************************************************

								 bessjp.c

Returns the derivative of J_n(x).

***************************************************************************/

#include "basic.h"
#include "../nr/nr.h"

double besskp( int n, double x)
{
  if (n==0)
	return -bessk(1,x);
  else
	return -0.5*(bessk(n-1,x)+bessk(n+1,x));
}
