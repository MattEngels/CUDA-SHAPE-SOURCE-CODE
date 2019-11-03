/*****************************************************************************************
                                                                                   head.h

Top-level header file for shape-cuda - adapted from head.h in v2.10.0 of shape.


*****************************************************************************************/
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>

#include "build.h"
#include "../nr/nr.h"
#include "../macros/files.h"
#include "../macros/func.h"
#include "../util/util.h"
#include "../astro/astro.h"
#include "../pgm/pgm.h"
#include "../shape/shape2.h"
#include <ctype.h>
#include "/home/matt/git/cfitsio/fitsio.h"
#include "const.h"
#include "../CudaCode/shape-cuda.h"
#include "../CudaUtils/cuda_utils.h"
#include "../CudaPR/parallel_reductions.h"
#include "../HostMultiThreading/hmt.h"
