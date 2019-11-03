/*  Returns the norm (magnitude) of a vector  */
/*  Added 2004 March 8 by CM                  */

#include "basic.h"

double vecnorm( double x[3])
{
  return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}
