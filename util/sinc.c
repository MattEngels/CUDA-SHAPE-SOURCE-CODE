#include "basic.h"

double sinc(double x)
{
  double y;

  if (fabs(x)<1.0e-10)
    return 1.0;
  y = x*3.141592653589793;
  y = sin(y)/y;
  return y;
}
