/***************************************************************************
                                                                normalize.c

Normalizes a vector to unit length.  Returns magnitude of original vector.

Modified 2009 July 24 by CM:
    Protect against division by zero for a zero-length vector
***************************************************************************/

#include "basic.h"

double normalize( double *u)
{
  int i;
  double norm;

  norm = 0.0;
  for (i=0; i<=2; i++)
    norm += u[i]*u[i];
  norm = sqrt(norm);
  if (norm != 0.0) {
    for (i=0; i<=2; i++)
      u[i] /= norm;
  }
  return norm;
}
