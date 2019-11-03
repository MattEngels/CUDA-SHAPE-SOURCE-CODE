extern "C" {
#include "../shape/head.h"
}
__device__ double dev_vecnorm( double x[3])
{
  return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}
