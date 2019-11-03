extern "C" {
#include "../shape/head.h"
}

__device__ int dev_vp_iround(double x)
{
  if (x < 0.0)
    return ((int)(x - 0.5));
  else
    return ((int)(x + 0.5));
}
__device__ int dev_vp_iroundf(float x)
{
  if (x < 0.0)
    return ((int)(x - 0.5));
  else
    return ((int)(x + 0.5));
}
