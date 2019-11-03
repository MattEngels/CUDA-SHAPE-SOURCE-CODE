extern "C" {
#include "../shape/head.h"
}
#include <stdio.h>

#define MAX_BLOCK_DIM_SIZE 65535
#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

static int deviceQueryFlag = 0;	/* This flag is set the first time getNumBlocksandThreads
* is used so that subsequent calls do not result in more deviceQuery calls */
static int maxTPB=0, maxGS[3];

int isPow2(unsigned int x){
    return ((x&(x-1))==0);
}
__device__ static double atomicMax64(double* address, double val){
	unsigned long long* address_as_i = (unsigned long long*) address;
	unsigned long long old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
				__double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}
unsigned int nextPow2(unsigned int x){
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
__inline__ __device__
double warpReduceSumD(double val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}
__inline__ __device__
double warpReduceMaxD(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val = fmaxf((__shfl_down(val, offset)),val);
  return val;
}
__inline__ __device__
double blockReduceSumD(double val) {
  static __shared__ double shared[32];
  int lane=threadIdx.x%warpSize;
  int wid=threadIdx.x/warpSize;
  val=warpReduceSumD(val);

  //write reduced value to shared memory
  if(lane==0) shared[wid]=val;
  __syncthreads();

  //ensure we only grab a value from shared memory if that warp existed
  val = (threadIdx.x<blockDim.x/warpSize) ? shared[lane] : double(0.0);
  if(wid==0) val=warpReduceSumD(val);

  return val;
}
__inline__ __device__ double blockReduceMaxD(double val) {
  static __shared__ double shared[32];
  int lane=threadIdx.x%warpSize;
  int wid=threadIdx.x/warpSize;
  val=warpReduceMaxD(val);

  //write reduced value to shared memory
  if(lane==0) shared[wid]=val;
  __syncthreads();

  //ensure we only grab a value from shared memory if that warp existed
  val = (threadIdx.x<blockDim.x/warpSize) ? shared[lane] : double(0.0);
  if(wid==0) val=warpReduceMaxD(val);

  return val;
}
__global__ void device_reduce_block_atomic_kernel(double *in, double* out, int N) {
  double sum=double(0.0);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    sum+=in[i];
  }
  sum=blockReduceSumD(sum);
  if(threadIdx.x==0)
    atomicAdd(out,sum);
}
__global__ void device_reduce_block_atomic_kernel_brt(struct pos_t **pos, double* out,
		int N, int f, int4 maxxylim) {
	/* Used for brightness calculation in light curves */
	double sum=double(0.0);
	int i, j, xspan;
	xspan = maxxylim.x - maxxylim.w + 1;//pos[f]->xlim[1] - pos[f]->xlim[0];

	for(int offset=blockIdx.x*blockDim.x+threadIdx.x; offset<N; offset+=blockDim.x*gridDim.x) {
		i = offset % xspan + maxxylim.w;//pos[f]->xlim[0];
		j = offset / xspan + maxxylim.y;//pos[f]->ylim[0];
		sum+=pos[f]->b[i][j];
	}
	sum=blockReduceSumD(sum);
	if(threadIdx.x==0)
		atomicAdd(&out[0],sum);
}
__global__ void device_reduce_block_atomic_kernel_o2(struct dat_t *ddat, double* out,
		int N, int f, int s) {
	/* Used for brightness calculation in light curves */
	double sum=double(0.0);
  int i, j;
  for(int offset=blockIdx.x*blockDim.x+threadIdx.x; offset<N; offset+=blockDim.x*gridDim.x) {
	  i = offset % ddat->set[s].desc.deldop.frame[f].ndel + 1;
	  j = offset / ddat->set[s].desc.deldop.frame[f].ndel + 1;

	  sum += ddat->set[s].desc.deldop.frame[f].obs[i][j] *
    		ddat->set[s].desc.deldop.frame[f].obs[i][j] *
    		ddat->set[s].desc.deldop.frame[f].oneovervar[i][j];
  }
  sum=blockReduceSumD(sum);
  if(threadIdx.x==0)
    atomicAdd_dbl(&out[0],sum);
}
__global__ void device_reduce_block_atomic_kernel_m2(struct dat_t *ddat, double* out,
		int N, int f, int s) {
	/* Used for brightness calculation in light curves */
  double sum=double(0.0);
  int i, j;
  for(int offset=blockIdx.x*blockDim.x+threadIdx.x; offset<N; offset+=blockDim.x*gridDim.x) {
	  i = offset % ddat->set[s].desc.deldop.frame[f].ndel + 1;
	  j = offset / ddat->set[s].desc.deldop.frame[f].ndel + 1;

	  sum += ddat->set[s].desc.deldop.frame[f].fit[i][j] *
    		ddat->set[s].desc.deldop.frame[f].fit[i][j] *
    		ddat->set[s].desc.deldop.frame[f].oneovervar[i][j];
  }
  sum=blockReduceSumD(sum);
  if(threadIdx.x==0)
    atomicAdd_dbl(&out[0],sum);
}
__global__ void device_reduce_block_atomic_kernel_om(struct dat_t *ddat, double* out,
		int N, int f, int s) {
	/* Used for brightness calculation in light curves */
  double sum=double(0.0);
  int i, j;
  for(int offset=blockIdx.x*blockDim.x+threadIdx.x; offset<N; offset+=blockDim.x*gridDim.x) {
	  i = offset % ddat->set[s].desc.deldop.frame[f].ndel + 1;
	  j = offset / ddat->set[s].desc.deldop.frame[f].ndel + 1;

	  sum += ddat->set[s].desc.deldop.frame[f].fit[i][j] *
    		ddat->set[s].desc.deldop.frame[f].obs[i][j] *
    		ddat->set[s].desc.deldop.frame[f].oneovervar[i][j];
  }
  sum=blockReduceSumD(sum);
  if(threadIdx.x==0)
    atomicAdd_dbl(&out[0],sum);
}
__global__ void device_zmax_block_atomic_krnl(struct pos_t **pos, double* out,
		int N, int f) {
  double zmax=float(0.0);
  int i, j, n;
  for (int offset=blockIdx.x*blockDim.x+threadIdx.x; offset<N; offset+=blockDim.x*gridDim.x) {
	  n = pos[f]->n;
	  i = offset % (2*n+1) - n;
	  j = offset / (2*n+1) - n;
	  zmax = fmaxf(zmax, pos[f]->z[i][j]);
  }
  zmax=blockReduceMaxD(zmax);
  if(threadIdx.x==0)
    atomicMax64(&out[0],zmax);
}
__global__ void device_reduce_block_atomic_kernel_ddf2(struct dat_t *ddat, double *out,
		int N, int f, int s) {
	/* Used for delay-doppler cross section calculation */
  double sum=0.0;
  int idel, idop;
  for(int offset=blockIdx.x * blockDim.x + threadIdx.x;
		  offset<N;
		  offset += blockDim.x * gridDim.x) {
        idel = offset % ddat->set[s].desc.deldop.frame[f].ndel + 1;
        idop = offset / ddat->set[s].desc.deldop.frame[f].ndel + 1;
        sum += ddat->set[s].desc.deldop.frame[f].fit[idel][idop];
  }
  sum=blockReduceSumD(sum);
  if(threadIdx.x==0)
    atomicAdd(out,sum);
}

/* Compute the number of threads and blocks to use for reduction kernel 6
 * For kernel 6, we observe the maximum specified number of blocks, because
 * each thread in that kernel can process a variable number of elements. */
float2 getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads)
{
	//get device capability, to avoid block/grid size exceed the upper bound
	cudaDeviceProp prop;
	int device, threads, blocks;
	float2 xb_yt;

	if (!deviceQueryFlag) { /* First time deviceQuery is called this run */

		gpuErrchk(cudaGetDevice(&device));
		gpuErrchk(cudaGetDeviceProperties(&prop, device));
		maxTPB = prop.maxThreadsPerBlock;
		maxGS[0] = prop.maxGridSize[0];
		maxGS[1] = prop.maxGridSize[1];
		maxGS[2] = prop.maxGridSize[2];
		deviceQueryFlag = 1;	/* Set flag so subsequent calls do not execute this again */
	}

	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);

	if ((float)threads*blocks > (float)maxGS[0] * maxTPB)
		printf("Array size for parallel reduction is too large!\n");

	if (blocks > maxGS[0])
	{
		printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
				blocks, maxGS[0], threads*2, threads);

		blocks /= 2;
		threads *= 2;
	}
	blocks = MIN(maxBlocks, blocks);
	xb_yt.x = blocks;
	xb_yt.y = threads;
	return xb_yt;
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, int nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, int nIsPow2>
__global__ void
maxz6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T myMax = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        myMax = fmaxf(g_idata[i], myMax);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            myMax = fmaxf(g_idata[i+blockSize], myMax);

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = myMax;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 256]);
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 128]);
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = myMax = fmaxf(myMax, sdata[tid +  64]);
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) myMax = fmaxf(sdata[tid + 32], myMax);
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            myMax = fmaxf(myMax, __shfl_down(myMax, offset));
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 32]);
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 16]);
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = myMax = fmaxf(myMax, sdata[tid +  8]);
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = myMax = fmaxf(myMax, sdata[tid +  4]);
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = myMax = fmaxf(myMax, sdata[tid +  2]);
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = myMax = fmaxf(myMax, sdata[tid +  1]);
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0)
    	g_odata[blockIdx.x] = myMax;
}
template <class T, unsigned int blockSize, int nIsPow2>
__global__ void
minz6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T myMin = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        myMin = fminf(g_idata[i], myMin);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            myMin = fminf(g_idata[i+blockSize], myMin);

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = myMin;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = myMin = fminf(myMin, sdata[tid + 256]);
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = myMin = fminf(myMin, sdata[tid + 128]);
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = myMin = fminf(myMin, sdata[tid +  64]);
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) myMin = fminf(sdata[tid + 32], myMin);
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            myMin = fminf(myMin, __shfl_down(myMin, offset));
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = myMin = fminf(myMin, sdata[tid + 32]);
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = myMin = fminf(myMin, sdata[tid + 16]);
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = myMin = fminf(myMin, sdata[tid +  8]);
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = myMin = fminf(myMin, sdata[tid +  4]);
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = myMin = fminf(myMin, sdata[tid +  2]);
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = myMin = fminf(myMin, sdata[tid +  1]);
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0)
    	g_odata[blockIdx.x] = myMin;
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduce(int size, int threads, int blocks,
       int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
        case 6:
        default:
            if (isPow2(size))
            {
                switch (threads)
                {
                    case 512:
                        reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 256:
                        reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 128:
                        reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 64:
                        reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 32:
                        reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 16:
                        reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  8:
                        reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  4:
                        reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  2:
                        reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  1:
                        reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;
                }
            }
            else
            {
                switch (threads)
                {
                    case 512:
                        reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 256:
                        reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 128:
                        reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 64:
                        reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 32:
                        reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 16:
                        reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  8:
                        reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  4:
                        reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  2:
                        reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  1:
                        reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;
                }
            }
            break;
    }
}

// Instantiate the reduction function for 3 types
template void
reduce<int>(int size, int threads, int blocks,
            int whichKernel, int *d_idata, int *d_odata);

template void
reduce<float>(int size, int threads, int blocks,
              int whichKernel, float *d_idata, float *d_odata);

template void
reduce<double>(int size, int threads, int blocks,
               int whichKernel, double *d_idata, double *d_odata);

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
maxz(int size, int threads, int blocks,
		int whichKernel, T *d_idata, T *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	// choose which of the optimized versions of reduction to launch
	switch (whichKernel)
	{
	case 6:
	default:
		if (isPow2(size))
		{
			switch (threads)
			{
			case 512:
				maxz6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 256:
				maxz6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 128:
				maxz6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 64:
				maxz6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 32:
				maxz6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 16:
				maxz6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  8:
				maxz6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  4:
				maxz6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  2:
				maxz6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  1:
				maxz6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;
			}
		}
		else
		{
			switch (threads)
			{
			case 512:
				maxz6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 256:
				maxz6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 128:
				maxz6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 64:
				maxz6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 32:
				maxz6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 16:
				maxz6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  8:
				maxz6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  4:
				maxz6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  2:
				maxz6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  1:
				maxz6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;
			}
		}
		break;
	}
}
// Instantiate the maxz function for 3 types
template void
maxz<int>(int size, int threads, int blocks,
		int whichKernel, int *d_idata, int *d_odata);

template void
maxz<float>(int size, int threads, int blocks,
              int whichKernel, float *d_idata, float *d_odata);

template void
maxz<double>(int size, int threads, int blocks,
               int whichKernel, double *d_idata, double *d_odata);


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
minz(int size, int threads, int blocks,
		int whichKernel, T *d_idata, T *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	// choose which of the optimized versions of reduction to launch
	switch (whichKernel)
	{
	case 6:
	default:
		if (isPow2(size))
		{
			switch (threads)
			{
			case 512:
				minz6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 256:
				minz6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 128:
				minz6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 64:
				minz6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 32:
				minz6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 16:
				minz6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  8:
				minz6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  4:
				minz6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  2:
				minz6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  1:
				minz6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;
			}
		}
		else
		{
			switch (threads)
			{
			case 512:
				minz6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 256:
				minz6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 128:
				minz6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 64:
				minz6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 32:
				minz6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case 16:
				minz6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  8:
				minz6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  4:
				minz6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  2:
				minz6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;

			case  1:
				minz6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
				break;
			}
		}
		break;
	}
}
// Instantiate the minz function for 3 types
template void
minz<int>(int size, int threads, int blocks,
		int whichKernel, int *d_idata, int *d_odata);

template void
minz<float>(int size, int threads, int blocks,
              int whichKernel, float *d_idata, float *d_odata);

template void
minz<double>(int size, int threads, int blocks,
               int whichKernel, double *d_idata, double *d_odata);

__global__ void set_idata_pntr_krnl(struct dat_t *ddat, double *d_idata,
		int set, int frm, int size) {
	/* MULTI-threaded kernel */
	int idel, idop, offset = blockIdx.x * blockDim.x + threadIdx.x;

	if (offset < size){

		switch (ddat->set[set].type) {
		case DELAY:
            idel = offset % ddat->set[set].desc.deldop.frame[frm].ndel + 1;
            idop = offset / ddat->set[set].desc.deldop.frame[frm].ndel + 1;
			d_idata[offset] = ddat->set[set].desc.deldop.frame[frm].fit[idel][idop];
			break;
		case DOPPLER:
			d_idata[offset] = ddat->set[set].desc.doppler.frame[frm].fit[offset+1];
			break;
		}
	}
}
__global__ void set_ab_penalty_terms_krnl(double *d_idata1, double *d_idata2,
		double *a, double *b, int size) {
	/* MULTI-threaded kernel */
	int offset = blockIdx.x * blockDim.x + threadIdx.x;

	if (offset < size){
		d_idata1[offset] = a[offset];
		d_idata2[offset] = b[offset];
	}
}
__global__ void set_single_double_array(double *d_idata, double *a, int size) {
	/* size-threaded kernel */
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		d_idata[i] = a[i];
	}
}
__global__ void set_idata_modarea_krnl(struct mod_t *dmod, double *d_idata,
		int c, int size) {
	/* MULTI-threaded kernel */
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
		d_idata[i] = dmod->shape.comp[c].real.f[i].area;
}
__global__ void set_dv_dcom_di_krnl(double *d_odata_dv, double *d_odata_dcom0,
		double *d_odata_dcom1, double *d_odata_dcom2, double *d_odata_dI00,
		double *d_odata_dI01, double *d_odata_dI02, double *d_odata_dI10,
		double *d_odata_dI11, double *d_odata_dI12, double *d_odata_dI20,
		double *d_odata_dI21, double *d_odata_dI22, struct mod_t *dmod, int c) {
	/* Single threaded kernel to update the model with volume, COM, and inertia */
	/* Note that this kernel ignores multi-threaded models for now */
	if (threadIdx.x == 0) {
		dmod->shape.comp[c].volume = dmod->shape.volume = d_odata_dv[0];
		dmod->shape.comp[c].com[0] = dmod->shape.com[0] = d_odata_dcom0[0];
		dmod->shape.comp[c].com[1] = dmod->shape.com[1] = d_odata_dcom1[0];
		dmod->shape.comp[c].com[2] = dmod->shape.com[2] = d_odata_dcom2[0];
		dmod->shape.comp[c].inertia[0][0] = dmod->shape.inertia[0][0] = d_odata_dI00[0];
		dmod->shape.comp[c].inertia[0][1] = dmod->shape.inertia[0][1] = d_odata_dI01[0];
		dmod->shape.comp[c].inertia[0][2] = dmod->shape.inertia[0][2] = d_odata_dI02[0];
		dmod->shape.comp[c].inertia[1][0] = dmod->shape.inertia[1][0] = d_odata_dI10[0];
		dmod->shape.comp[c].inertia[1][1] = dmod->shape.inertia[1][1] = d_odata_dI11[0];
		dmod->shape.comp[c].inertia[1][2] = dmod->shape.inertia[1][2] = d_odata_dI12[0];
		dmod->shape.comp[c].inertia[2][0] = dmod->shape.inertia[2][0] = d_odata_dI20[0];
		dmod->shape.comp[c].inertia[2][1] = dmod->shape.inertia[2][1] = d_odata_dI21[0];
		dmod->shape.comp[c].inertia[2][2] = dmod->shape.inertia[2][2] = d_odata_dI22[0];
	}
}
__global__ void set_lghtcrv_values_krnl(struct dat_t *ddat, int s, double *d_odata,
		int i) {
	/* Single-threaded kernel (but streamed) */
	if (threadIdx.x ==0) {
		ddat->set[s].desc.lghtcrv.y[i] = d_odata[0];
	}
}
__global__ void set_o2m2om_values_krnl(double *d_odata0, double *d_odata1,
		double *d_odata2, double *o2, double *om, double *m2, int f) {

	/* Single-threaded kernel (but streamed) */
	if (threadIdx.x ==0) {
		o2[f] += d_odata0[0];
		m2[f] += d_odata1[0];
		om[f] += d_odata2[0];
	}
}
__global__ void zmax_intermediary_krnl(struct dat_t *ddat, double *d_in,
		double *d_out, int s, int f) {
	/* Single-threaded kernel to add each frame's zmax to a sum for the set */
	if (threadIdx.x ==0)
		d_out[f] = d_in[0] * ddat->set[s].desc.deldop.frame[f].weight;
}
__global__ void zmax_final_krnl(double *d_odata_final, int nframes) {
	/* Single threaded kernel to sup up the zmax figures for all frames */

	if (threadIdx.x == 0)
		for (int f=1; f<nframes; f++)
			d_odata_final[0] += d_odata_final[f];
}
__global__ void deldop_xsec_intermediate_krnl(struct dat_t *ddat, double *d_odata,
		double *deldop_cross_section, double *dsum_rad_xsec, int s, int f) {
	if (threadIdx.x==0) {
		deldop_cross_section[f] = ddat->set[s].desc.deldop.frame[f].overflow_xsec;
		deldop_cross_section[f] += d_odata[0];
		deldop_cross_section[f] *= ddat->set[s].desc.deldop.frame[f].cal.val;
		atomicAdd(&dsum_rad_xsec[0], (ddat->set[s].desc.deldop.frame[f].weight *
				deldop_cross_section[f]));
	}
}

__host__ void sum_brightness_gpu(struct dat_t *ddat, struct pos_t **pos,
		int nframes, int size, int flt, int set, int maxthds,
		int4 maxxylim, cudaStream_t *sb_stream) {
	/* Function sums up all array elements in the pos[i]->b[][] array.
	 * The output for each individual pos is what used to be calculated by
	 * apply_photo alone  */

	/* This version gets all frames in the set done simultaneously in streams,
	 * which are passed to this function from the parent function.
	 *
	 * Assumptions made for this streams version:
	 * 	- All pos in the set have the same parameters (i.e., size)
	 *
	 * Input arguments:
	 * 	- pos		- all pos structures in current dataset
	 * 	- nframes	- number of frames in dataset
	 * 	- size		- size of each pos array, i.e. # of pixels
	 * 	- flt		- flag wether to use a float reduction or not
	 * 	- sb_stream	- array of cudaStreams, guaranteed to be as large as the
	 * 				  largest dataset requires
	 * */
	size = maxthds;
	int f=0, maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	double *d_odata0, *d_odata1, *d_odata2, *d_odata3, *d_odata4, *d_odata5;
	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads
	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock;
	BLK.x = floor((THD.x - 1 + size)/THD.x);

	/* Find number of blocks & threads needed for reduction call */
	/* NOTE: This assumes all frames/pos are the same size!      */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	size_t arrsz = sizeof(double) * numBlocks;

	/* Allocate memory for d_odata */
	gpuErrchk(cudaMalloc((void**)&d_odata0, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata1, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata2, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata3, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata4, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata5, arrsz));

	cudaMemsetAsync(d_odata0, 0, arrsz, sb_stream[0]);
	cudaMemsetAsync(d_odata1, 0, arrsz, sb_stream[1]);
	cudaMemsetAsync(d_odata2, 0, arrsz, sb_stream[2]);
	cudaMemsetAsync(d_odata3, 0, arrsz, sb_stream[3]);
	cudaMemsetAsync(d_odata4, 0, arrsz, sb_stream[4]);
	cudaMemsetAsync(d_odata5, 0, arrsz, sb_stream[5]);

	/* Call reduction  */
	while (f<=nframes) {
		f++;
		if (f > nframes) break;
		device_reduce_block_atomic_kernel_brt<<< dimGrid,dimBlock,0,sb_stream[0]>>>
				(pos, d_odata0, size, f, maxxylim);
		set_lghtcrv_values_krnl<<<1,1,0,sb_stream[0]>>>(ddat, set, d_odata0, f);
		cudaMemsetAsync(d_odata0, 0, arrsz, sb_stream[0]);

		f++;
		if (f > nframes) break;
		device_reduce_block_atomic_kernel_brt<<< dimGrid,dimBlock,0,sb_stream[1]>>>
				(pos, d_odata1, size, f, maxxylim);
		set_lghtcrv_values_krnl<<<1,1,0,sb_stream[1]>>>(ddat, set, d_odata1, f);
		cudaMemsetAsync(d_odata1, 0, arrsz, sb_stream[1]);

		f++;
		if (f > nframes) break;
		device_reduce_block_atomic_kernel_brt<<< dimGrid,dimBlock,0,sb_stream[2]>>>
				(pos, d_odata2, size, f, maxxylim);
		set_lghtcrv_values_krnl<<<1,1,0,sb_stream[2]>>>(ddat, set, d_odata2, f);
		cudaMemsetAsync(d_odata2, 0, arrsz, sb_stream[2]);

		f++;
		if (f > nframes) break;
		device_reduce_block_atomic_kernel_brt<<< dimGrid,dimBlock,0,sb_stream[3]>>>
				(pos, d_odata3, size, f, maxxylim);
		set_lghtcrv_values_krnl<<<1,1,0,sb_stream[3]>>>(ddat, set, d_odata3, f);
		cudaMemsetAsync(d_odata3, 0, arrsz, sb_stream[3]);

		f++;
		if (f > nframes) break;
		device_reduce_block_atomic_kernel_brt<<< dimGrid,dimBlock,0,sb_stream[4]>>>
				(pos, d_odata4, size, f, maxxylim);
		set_lghtcrv_values_krnl<<<1,1,0,sb_stream[4]>>>(ddat, set, d_odata4, f);
		cudaMemsetAsync(d_odata4, 0, arrsz, sb_stream[4]);

		f++;
		if (f > nframes) break;
		device_reduce_block_atomic_kernel_brt<<< dimGrid,dimBlock,0,sb_stream[5]>>>
				(pos, d_odata5, size, f, maxxylim);
		set_lghtcrv_values_krnl<<<1,1,0,sb_stream[5]>>>(ddat, set, d_odata5, f);
		cudaMemsetAsync(d_odata5, 0, arrsz, sb_stream[5]);
	}
	checkErrorAfterKernelLaunch("device_reduce_block_atomic_krnl_brt");
//	for (f=0; f<6; f++)
//		cudaStreamSynchronize(sb_stream[f]);


	cudaFree(d_odata0);
	cudaFree(d_odata1);
	cudaFree(d_odata2);
	cudaFree(d_odata3);
	cudaFree(d_odata4);
	cudaFree(d_odata5);
}

__host__ double compute_zmax_gpu(struct dat_t *ddat, struct pos_t **pos,
		int nframes, int size, int set, cudaStream_t *sb_stream) {
	/* Function finds zmax for each deldop frame (pos[f]->z[]) in a streamed
	 * calculation. Eight frames at a time.
	 * This version gets all frames in the set done simultaneously in streams,
	 * which are passed to this function from the parent function.
	 *
	 * Assumptions made for this streams version:
	 * 	- All pos in the set have the same parameters (i.e., size)
	 *
	 * Input arguments:
	 * 	- pos		- all pos structures in current dataset
	 * 	- nframes	- number of frames in dataset
	 * 	- size		- size of each pos array, i.e. # of pixels
	 * 	- sb_stream	- array of cudaStreams, guaranteed to be as large as the
	 * 				  largest dataset requires and at least 13	  */

	int f=-1, maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	double *d_odata0, *d_odata1, *d_odata2, *d_odata3, *d_odata4, *d_odata_final,
        *sum_deldop_zmax_arr, sum_deldop_zmax;
	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads
	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock;
	BLK.x = floor((THD.x - 1 + size)/THD.x);

	/* Find number of blocks & threads needed for reduction call */
	/* NOTE: This assumes all frames/pos are the same size!      */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	size_t arrsz = sizeof(double) * numBlocks;
	size_t arrsz2 = sizeof(double) * nframes;

//	/* Synchronize streams to default stream */
//	for (int f=0; f<nframes; f++)
//		cudaStreamSynchronize(sb_stream[f]);

	/* Allocate memory */
	sum_deldop_zmax_arr = (double *)malloc(arrsz2);
	gpuErrchk(cudaMalloc((void**)&d_odata0, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata1, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata2, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata3, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata4, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_final, arrsz2));
	gpuErrchk(cudaMemsetAsync(d_odata_final, 0, arrsz2, sb_stream[8]));

	/* Call reduction  */
	while (f<nframes) {
		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata0, 0, arrsz, sb_stream[0]);
		device_zmax_block_atomic_krnl<<< dimGrid,dimBlock,0,sb_stream[0]>>>
				(pos, d_odata0,	size, f);
		zmax_intermediary_krnl<<<1,1,0,sb_stream[0]>>>
				(ddat, d_odata0, d_odata_final, set, f);

		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata1, 0, arrsz, sb_stream[1]);
		device_zmax_block_atomic_krnl<<< dimGrid,dimBlock,0,sb_stream[1]>>>
				(pos, d_odata1, size, f);
		zmax_intermediary_krnl<<<1,1,0,sb_stream[1]>>>
				(ddat, d_odata1, d_odata_final, set, f);

		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata2, 0, arrsz, sb_stream[2]);
		device_zmax_block_atomic_krnl<<< dimGrid,dimBlock,0,sb_stream[2]>>>
				(pos, d_odata2, size, f);
		zmax_intermediary_krnl<<<1,1,0,sb_stream[2]>>>
				(ddat, d_odata2, d_odata_final, set, f);

		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata3, 0, arrsz, sb_stream[3]);
		device_zmax_block_atomic_krnl<<< dimGrid,dimBlock,0,sb_stream[3]>>>
				(pos, d_odata3, size, f);
		zmax_intermediary_krnl<<<1,1,0,sb_stream[3]>>>
				(ddat, d_odata3, d_odata_final, set, f);

		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata4, 0, arrsz, sb_stream[4]);
		device_zmax_block_atomic_krnl<<< dimGrid,dimBlock,0,sb_stream[4]>>>
				(pos, d_odata4, size, f);
		zmax_intermediary_krnl<<<1,1,0,sb_stream[4]>>>
				(ddat, d_odata4, d_odata_final, set, f);
	}
	checkErrorAfterKernelLaunch("device_zmax_block_atomic_krnl");

//	for (f=0; f<5; f++)
//		cudaStreamSynchronize(sb_stream[f]);

	/* Now just sum up all frames in d_odata_final in a loop, copy results back
	 * to a host array and then return the first element - sum_deldop_zmax */
	zmax_final_krnl<<<1,1>>>(d_odata_final, nframes);
	checkErrorAfterKernelLaunch("zmax_final_krnl");
	gpuErrchk(cudaMemcpy(sum_deldop_zmax_arr, d_odata_final, arrsz2,
			cudaMemcpyDeviceToHost));
	sum_deldop_zmax = sum_deldop_zmax_arr[0];

	free(sum_deldop_zmax_arr);
	cudaFree(d_odata0);
	cudaFree(d_odata1);
	cudaFree(d_odata2);
	cudaFree(d_odata3);
	cudaFree(d_odata4);
	cudaFree(d_odata_final);
	return sum_deldop_zmax;
}

__host__ double compute_deldop_xsec_gpu(struct dat_t *ddat, int nframes,
		int size, int set, cudaStream_t *sb_stream) {
	/* Function finds zmax for each deldop frame (pos[f]->z[]) in a streamed
	 * calculation. Eight frames at a time.
	 * This version gets all frames in the set done simultaneously in streams,
	 * which are passed to this function from the parent function.
	 *
	 * Assumptions made for this streams version:
	 * 	- All pos in the set have the same parameters (i.e., size)
	 *
	 * Input arguments:
	 * 	- pos		- all pos structures in current dataset
	 * 	- nframes	- number of frames in dataset
	 * 	- size		- size of Delay-Doppler space (ndop*ndel)
	 * 	- sb_stream	- array of cudaStreams, guaranteed to be as large as the
	 * 				  largest dataset requires and at least 13	  */

	int f=-1, maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	double *d_odata0, *d_odata1, *d_odata2, *d_odata3, *d_odata4, sum_deldop_xsec;
	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads
	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock;
	BLK.x = floor((THD.x - 1 + size)/THD.x);
	double *deldop_cross_section, *dsum_rad_xsec, *hsum_rad_xsec;
	/* Find number of blocks & threads needed for reduction call */
	/* NOTE: This assumes all frames/pos are the same size!      */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	size_t arrsz = sizeof(double) * numBlocks;
	size_t arrsz2 = sizeof(double) * nframes;

	/* Allocate memory */
	hsum_rad_xsec = (double *)malloc(sizeof(double)*2);
	gpuErrchk(cudaMalloc((void**)&dsum_rad_xsec, sizeof(double)*2));
	gpuErrchk(cudaMalloc((void**)&d_odata0, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata1, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata2, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata3, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata4, arrsz));
	gpuErrchk(cudaMalloc((void**)&deldop_cross_section, arrsz2));
	gpuErrchk(cudaMemset(deldop_cross_section, 0, arrsz2));
	gpuErrchk(cudaMemset(dsum_rad_xsec, 0, sizeof(double)*2));

	/* Call reduction  */
	while (f<nframes) {
		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata0, 0, arrsz, sb_stream[0]);
		device_reduce_block_atomic_kernel_ddf2<<< dimGrid, dimBlock, 0, sb_stream[0]>>>
				(ddat,d_odata0, size, f, set);
		deldop_xsec_intermediate_krnl<<<1,1,0,sb_stream[0]>>>(ddat, d_odata0,
				deldop_cross_section, dsum_rad_xsec, set, f);
		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata1, 0, arrsz, sb_stream[1]);
		device_reduce_block_atomic_kernel_ddf2<<< dimGrid, dimBlock, 0, sb_stream[1]>>>
				(ddat, d_odata1, size, f, set);
		deldop_xsec_intermediate_krnl<<<1,1,0,sb_stream[1]>>>(ddat, d_odata1,
				deldop_cross_section, dsum_rad_xsec, set, f);

		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata2, 0, arrsz, sb_stream[2]);
		device_reduce_block_atomic_kernel_ddf2<<< dimGrid, dimBlock, 0, sb_stream[2]>>>
				(ddat, d_odata2, size, f, set);
		deldop_xsec_intermediate_krnl<<<1,1,0,sb_stream[2]>>>(ddat, d_odata2,
				deldop_cross_section, dsum_rad_xsec, set, f);

		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata3, 0, arrsz, sb_stream[3]);
		device_reduce_block_atomic_kernel_ddf2<<< dimGrid, dimBlock, 0, sb_stream[3]>>>
				(ddat, d_odata3, size, f, set);
		deldop_xsec_intermediate_krnl<<<1,1,0,sb_stream[3]>>>(ddat, d_odata3,
				deldop_cross_section, dsum_rad_xsec, set, f);

		f++;
		if (f >= nframes) break;
		cudaMemsetAsync(d_odata4, 0, arrsz, sb_stream[4]);
		device_reduce_block_atomic_kernel_ddf2<<< dimGrid, dimBlock, 0, sb_stream[4]>>>
				(ddat, d_odata4, size, f, set);
		deldop_xsec_intermediate_krnl<<<1,1,0,sb_stream[4]>>>(ddat, d_odata4,
				deldop_cross_section, dsum_rad_xsec, set, f);
	}
	checkErrorAfterKernelLaunch("device_reduce_block_atomic_krnl_ddf2");

//	for (f=0; f<5; f++)
//		cudaStreamSynchronize(sb_stream[f]);

	/* Now copy dsum_rad_xsec over to a host copy */
	gpuErrchk(cudaMemcpy(hsum_rad_xsec, dsum_rad_xsec, sizeof(double)*2,
			cudaMemcpyDeviceToHost));
	sum_deldop_xsec = hsum_rad_xsec[0];

	free(hsum_rad_xsec);
	cudaFree(dsum_rad_xsec);
	cudaFree(deldop_cross_section);
	cudaFree(d_odata0);
	cudaFree(d_odata1);
	cudaFree(d_odata2);
	cudaFree(d_odata3);
	cudaFree(d_odata4);
	return sum_deldop_xsec;
}

__host__ void sum_2_double_arrays(double *a, double *b, double *absum, int size) {
	/* Function sums up two arrays of doubles.  Returns both sums in host-array
	 * absum  */

	int s;
	int maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int whichKernel = 6;		// id of reduction kernel
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	double *d_odata1;				// temp. float array for reduction output
	double *d_odata2;
	double *d_idata1; 			// temp. float arrays for reduction input
	double *d_idata2;
	double *h_odata1;				// the host output array
	double *h_odata2;
	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads

	dim3 BLK,THD;
	BLK.x = floor((maxThreadsPerBlock - 1 + size)/maxThreadsPerBlock);
	THD.x = maxThreadsPerBlock; // Thread block dimensions

	/* Allocate memory for d_idata, then set that pointer equal to the right
	 * data set and frame to the right deldop fit array	 */
	gpuErrchk(cudaMalloc((void**)&d_idata1, sizeof(double) * size));
	gpuErrchk(cudaMalloc((void**)&d_idata2, sizeof(double) * size));

	set_ab_penalty_terms_krnl<<<BLK,THD>>>(d_idata1, d_idata2,
			a, b, size);
	checkErrorAfterKernelLaunch("set_ab_penalty_terms_krnl");

	/* Find number of blocks & threads needed for reduction call. Note that
	 * both a and b reductions use the same parameters for size (because both
	 * arrays are of the same size!) */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;

	/* Allocate memory for d_odata and d_odata2 with enough elements to hold
	 * the reduction of each block during the first call */
	gpuErrchk(cudaMalloc((void**)&d_odata1, sizeof(double) * numBlocks));
	gpuErrchk(cudaMalloc((void**)&d_odata2, sizeof(double) * numBlocks));
	h_odata1 = (double *)malloc(numBlocks * sizeof(double));
	h_odata2 = (double *)malloc(numBlocks * sizeof(double));

	/* Call reduction for first time on both arrays */
	reduce<double>(size, numThreads, numBlocks, whichKernel, d_idata1, d_odata1);
	reduce<double>(size, numThreads, numBlocks, whichKernel, d_idata2, d_odata2);
	checkErrorAfterKernelLaunch("reduce<double> in sum_a_b_penalty_terms");

	/* Reset d_idata for later use as buffer */
	cudaMemset(d_idata1, 0, size*sizeof(double));
	cudaMemset(d_idata2, 0, size*sizeof(double));

	/* Now sum partial block sums on GPU, using the reduce6<> kernel */
	s = numBlocks;

	while (s > 1)
	{
		int threads = 0, blocks = 0;
		xblock_ythread = getNumBlocksAndThreads(s, maxBlocks, maxThreads);
		blocks = xblock_ythread.x;
		threads = xblock_ythread.y;

		/* Copy the first d_odata back into d_idata2  */
		cudaMemcpy(d_idata1, d_odata1, s*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_idata2, d_odata2, s*sizeof(double), cudaMemcpyDeviceToDevice);

		reduce<double>(s, threads, blocks, whichKernel, d_idata1, d_odata1);
		reduce<double>(s, threads, blocks, whichKernel, d_idata2, d_odata2);

		if (whichKernel < 3)
			s = (s + threads - 1) / threads;
		else
			s = (s + (threads*2-1)) / (threads*2);
		if (s > 1)
			printf("s is bigger than one");
	}

	gpuErrchk(cudaMemcpy(h_odata1, d_odata1, 2*sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_odata2, d_odata2, 2*sizeof(double), cudaMemcpyDeviceToHost));
	absum[0] = h_odata1[0]; /* a sum */
	absum[1] = h_odata2[0]; /* b sum */
	free(h_odata1);
	free(h_odata2);
	cudaFree(d_odata1);
	cudaFree(d_odata2);
	cudaFree(d_idata1);
	cudaFree(d_idata2);
}

__host__ double sum_double_array(double *a, int size) {
	/* Function sums up two arrays of doubles.  Returns both sums in host-array
	 * absum  */

	int s;
	int maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int whichKernel = 6;		// id of reduction kernel
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	double out = 0.0;
	double *d_odata;				// temp. float array for reduction output
	double *d_idata; 			// temp. float arrays for reduction input
	double *h_odata;				// the host output array
	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads

	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock; // Thread block dimensions
	BLK.x = floor((THD.x - 1 + size)/THD.x);

	/* Allocate memory for d_idata, then set that pointer equal to the right
	 * data set and frame to the right deldop fit array	 */
	gpuErrchk(cudaMalloc((void**)&d_idata, sizeof(double) * size));

	set_single_double_array<<<BLK,THD>>>(d_idata, a, size);
	checkErrorAfterKernelLaunch("set_single_double_array_krnl");

	/* Find number of blocks & threads needed for reduction call. Note that
	 * both a and b reductions use the same parameters for size (because both
	 * arrays are of the same size!) */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;

	/* Allocate memory for d_odata and d_odata2 with enough elements to hold
	 * the reduction of each block during the first call */
	gpuErrchk(cudaMalloc((void**)&d_odata, sizeof(double) * numBlocks));
	h_odata = (double *)malloc(numBlocks * sizeof(double));

	/* Call reduction for first time on both arrays */
	reduce<double>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);
	checkErrorAfterKernelLaunch("reduce<double> in sum_a_b_penalty_terms");

	/* Reset d_idata for later use as buffer */
	cudaMemset(d_idata, 0, size*sizeof(double));

	/* Now sum partial block sums on GPU, using the reduce6<> kernel */
	s = numBlocks;

	while (s > 1)
	{
		int threads = 0, blocks = 0;
		xblock_ythread = getNumBlocksAndThreads(s, maxBlocks, maxThreads);
		blocks = xblock_ythread.x;
		threads = xblock_ythread.y;

		/* Copy the first d_odata back into d_idata2  */
		cudaMemcpy(d_idata, d_odata, s*sizeof(double), cudaMemcpyDeviceToDevice);

		reduce<double>(s, threads, blocks, whichKernel, d_idata, d_odata);

		if (whichKernel < 3)
			s = (s + threads - 1) / threads;
		else
			s = (s + (threads*2-1)) / (threads*2);
		if (s > 1)
			printf("s is bigger than one");
	}

	gpuErrchk(cudaMemcpy(h_odata, d_odata, 2*sizeof(double), cudaMemcpyDeviceToHost));
	out = h_odata[0];

	free(h_odata);
	cudaFree(d_odata);
	cudaFree(d_idata);
	return out;
}

__host__ double find_max_in_double_array(double *in, int size) {
	/* Function calculates the zmax in a pos->z arrary with
	 * Nvidia's reduction sample code (simplified and adapted for use with
	 * shape).  The function returns the cross section as a float 	 */

	int s;						// array size
	int maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int whichKernel = 6;		// id of reduction kernel
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	double max = 0.0;			// radar cross section; return value
	double *d_odata;				// temp. float array for reduction output
	double *d_idata; 			// temp. float arrays for reduction input
	double *h_odata;				// the host output array
	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads

	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock;
	BLK.x = floor((THD.x - 1 + size)/THD.x);

	/* Allocate memory for d_idata, then set that pointer equal to the right
	 * data set and frame to the right deldop fit array	 */
	cudaCalloc((void**)&d_idata, sizeof(double), size);

	set_single_double_array<<<BLK,THD>>>(d_idata, in, size);
	checkErrorAfterKernelLaunch("set_single_double_array");

	/* Find number of blocks & threads needed for reduction call */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;

	/* Allocate memory for d_odata and d_odata2 with enough elements to hold
	 * the reduction of each block during the first call */
	cudaCalloc((void**)&d_odata,  sizeof(double), numBlocks);
	h_odata = (double *) malloc(numBlocks*sizeof(double));

	/* Call maxz for first time */
	maxz<double>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);
	checkErrorAfterKernelLaunch("maxz<double> in find_max_in_double_array");

	/* Reset d_idata for later use as buffer */
	cudaMemset(d_idata, 0, size*sizeof(double));

	/* Now sum partial block sums on GPU, using the maxz6<> kernel */
	s = numBlocks;

	while (s > 1)
	{
		int threads = 0, blocks = 0;
		xblock_ythread = getNumBlocksAndThreads(s, maxBlocks, maxThreads);
		blocks = xblock_ythread.x;
		threads = xblock_ythread.y;

		/* Copy the first d_odata back into d_idata2  */
		cudaMemcpy(d_idata, d_odata, s*sizeof(double), cudaMemcpyDeviceToDevice);

		maxz<double>(s, threads, blocks, whichKernel, d_idata, d_odata);

		if (whichKernel < 3)
			s = (s + threads - 1) / threads;
		else
			s = (s + (threads*2-1)) / (threads*2);
		if (s > 1)
			printf("s is bigger than one");
	}

	gpuErrchk(cudaMemcpy(h_odata, d_odata, 2*sizeof(double), cudaMemcpyDeviceToHost));
	max = h_odata[0];
	free(h_odata);
	cudaFree(d_odata);
	cudaFree(d_idata);
	return max;
}

__host__ double find_min_in_double_array(double *in, int size) {
	/* Function calculates the zmax in a pos->z arrary with
	 * Nvidia's reduction sample code (simplified and adapted for use with
	 * shape).  The function returns the cross section as a float 	 */

	int s;						// array size
	int maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int whichKernel = 6;		// id of reduction kernel
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	double min = 0.0;			// radar cross section; return value
	double *d_odata;				// temp. float array for reduction output
	double *d_idata; 			// temp. float arrays for reduction input
	double *h_odata;				// the host output array
	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads

	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock;
	BLK.x = floor((THD.x - 1 + size)/THD.x);

	/* Allocate memory for d_idata, then set that pointer equal to the right
	 * data set and frame to the right deldop fit array	 */
	cudaCalloc((void**)&d_idata, sizeof(double), size);

	set_single_double_array<<<BLK,THD>>>(d_idata, in, size);
	checkErrorAfterKernelLaunch("set_single_double_array");

	/* Find number of blocks & threads needed for reduction call */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;

	/* Allocate memory for d_odata and d_odata2 with enough elements to hold
	 * the reduction of each block during the first call */
	cudaCalloc((void**)&d_odata,  sizeof(double), numBlocks);
	h_odata = (double *) malloc(numBlocks*sizeof(double));

	/* Call maxz for first time */
	minz<double>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);
	checkErrorAfterKernelLaunch("minz<double> in find_min_in_double_array");

	/* Reset d_idata for later use as buffer */
	cudaMemset(d_idata, 0, size*sizeof(double));

	/* Now sum partial block sums on GPU, using the maxz6<> kernel */
	s = numBlocks;

	while (s > 1)
	{
		int threads = 0, blocks = 0;
		xblock_ythread = getNumBlocksAndThreads(s, maxBlocks, maxThreads);
		blocks = xblock_ythread.x;
		threads = xblock_ythread.y;

		/* Copy the first d_odata back into d_idata2  */
		cudaMemcpy(d_idata, d_odata, s*sizeof(double), cudaMemcpyDeviceToDevice);

		minz<double>(s, threads, blocks, whichKernel, d_idata, d_odata);

		if (whichKernel < 3)
			s = (s + threads - 1) / threads;
		else
			s = (s + (threads*2-1)) / (threads*2);
		if (s > 1)
			printf("s is bigger than one");
	}

	gpuErrchk(cudaMemcpy(h_odata, d_odata, 2*sizeof(double), cudaMemcpyDeviceToHost));
	min = h_odata[0];
	free(h_odata);
	cudaFree(d_odata);
	cudaFree(d_idata);
	return min;
}

__host__ double compute_doppler_xsec(struct dat_t *ddat, int ndop,
		int set, int frm) {
	/* Function calculates a delay-Doppler frame's radar cross section with
	 * Nvidia's reduction sample code (simplified and adapted for use with
	 * shape).  The function returns the cross section as a float 	 */

	int s, size=ndop;			// array size
	int maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int whichKernel = 6;		// id of reduction kernel
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	double xsec = 0.0;			// radar cross section; return value
	double *d_odata;				// temp. float array for reduction output
	double *d_idata; 			// temp. float arrays for reduction input
	double *h_odata;				// the host output array
	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads

	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock;
	BLK.x = floor((THD.x - 1 + size)/THD.x);

	/* Allocate memory for d_idata, then set that pointer equal to the right
	 * data set and frame to the right deldop fit array	 */
	cudaCalloc((void**)&d_idata, sizeof(double), size);

	set_idata_pntr_krnl<<<BLK,THD>>>(ddat, d_idata, set, frm, size);
	checkErrorAfterKernelLaunch("set_idata_pntr_krnl64 in compute_doppler_xsec64");

	/* Find number of blocks & threads needed for reduction call */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;

	/* Allocate memory for d_odata and d_odata2 with enough elements to hold
	 * the reduction of each block during the first call */
	cudaCalloc((void**)&d_odata,  sizeof(double), numBlocks);
	h_odata = (double *) malloc(numBlocks*sizeof(double));

	/* Call reduction for first time */
	reduce<double>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);
	checkErrorAfterKernelLaunch("reduce<double> in compute_deldop_xsec");

	/* Reset d_idata for later use as buffer */
	gpuErrchk(cudaMemset(d_idata, 0, size*sizeof(double)));

	/* Now sum partial block sums on GPU, using the reduce6<> kernel */
	s = numBlocks;

	while (s > 1)
	{
		int threads = 0, blocks = 0;
		xblock_ythread = getNumBlocksAndThreads(s, maxBlocks, maxThreads);
		blocks = xblock_ythread.x;
		threads = xblock_ythread.y;

		/* Copy the first d_odata back into d_idata2  */
		cudaMemcpy(d_idata, d_odata, s*sizeof(double), cudaMemcpyDeviceToDevice);

		reduce<double>(s, threads, blocks, whichKernel, d_idata, d_odata);

		if (whichKernel < 3)
			s = (s + threads - 1) / threads;
		else
			s = (s + (threads*2-1)) / (threads*2);
		if (s > 1)
			printf("s is bigger than one");
	}

	gpuErrchk(cudaMemcpy(h_odata, d_odata, 1*sizeof(double), cudaMemcpyDeviceToHost));
	xsec = h_odata[0];
	free(h_odata);
	cudaFree(d_odata);
	cudaFree(d_idata);
	return xsec;
}

__host__ double compute_model_area(struct mod_t *dmod, int c, int size) {
	/* Function calculates the model's surface area with Nvidia's reduction
	 * sample code (simplified and adapted for use with shape).  The function
	 * returns the cross section as a double. There is another similar function
	 * which returns a float 	 */
	int maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	double area = 0.0;			// radar cross section; return value
	double *d_odata, *d_idata, *h_data;

	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads

	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock; // Thread block dimensions
	BLK.x = floor((THD.x - 1 + size)/THD.x);

	/* Find number of blocks & threads needed for reduction call */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);

	/* Allocate memory for d_idata and d_odata */
	gpuErrchk(cudaMalloc((void**)&d_idata, sizeof(double) * size));
	gpuErrchk(cudaMalloc((void**)&d_odata,  sizeof(double)* numBlocks));
	gpuErrchk(cudaMemset(d_odata, 0, sizeof(double)*numBlocks));
	h_data = (double *) malloc(numBlocks*sizeof(double));

	/* Load the d_idata array */
	set_idata_modarea_krnl<<<BLK,THD>>>(dmod, d_idata, c, size);
	checkErrorAfterKernelLaunch("set_idata_modarea_krnl");

	/* Call reduction and copy back to host */
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock>>>(d_idata, d_odata, size);
	checkErrorAfterKernelLaunch("device_reduce_block_atomic_kernel");
	gpuErrchk(cudaMemcpy(h_data, d_odata, sizeof(double)*numBlocks, cudaMemcpyDeviceToHost));

	area = h_data[0];
	cudaFree(d_odata);
	cudaFree(d_idata);
	free(h_data);
	return area;
}

__host__ void dvdI_reduce_gpu(struct mod_t *dmod, double *dv, double *dcom0,
		double *dcom1, double *dcom2, double *dI00, double *dI01, double *dI02,
		double *dI10, double *dI11, double *dI12, double *dI20, double *dI21,
		double *dI22, int size, int c, cudaStream_t *dv_streams)
{
	int maxThreads = maxThreadsPerBlock;		// max # of threads per block
	int maxBlocks = 2048;		// max # of blocks per grid
	int numBlocks = 0;			// initialize numBlocks
	int numThreads = 0;			// initialize numThreads
	float2 xblock_ythread;		// used for return value of getNumBlocksAndThreads
	/* Device output arrays */
	double *d_odata_dcom0, *d_odata_dcom1, *d_odata_dcom2, *d_odata_dI00,
	*d_odata_dI01, *d_odata_dI02, *d_odata_dI10, *d_odata_dI11,
	*d_odata_dI12, *d_odata_dI20, *d_odata_dI21, *d_odata_dI22,
	*d_odata_dv;

	/* Find number of blocks & threads needed for reduction call */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	size_t arrsz = sizeof(double)*numBlocks;
	
	/* Allocate memory for the device output arrays for first reduction */
	gpuErrchk(cudaMalloc((void**)&d_odata_dv,    arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dcom0, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dcom1, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dcom2, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dI00,	 arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dI01,  arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dI02,  arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dI10,  arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dI11,  arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dI12,  arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dI20,  arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dI21,  arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata_dI22,  arrsz));

	gpuErrchk(cudaMemsetAsync(d_odata_dv, 0, arrsz, dv_streams[0]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[0]>>>(dv, d_odata_dv, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dcom0, 0, arrsz, dv_streams[1]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[1]>>>(dcom0, d_odata_dcom0, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dcom1, 0, arrsz, dv_streams[2]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[2]>>>(dcom1, d_odata_dcom1, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dcom2, 0, arrsz, dv_streams[3]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[3]>>>(dcom2, d_odata_dcom2, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dI00, 0, arrsz, dv_streams[4]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[4]>>>(dI00, d_odata_dI00, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dI01, 0, arrsz, dv_streams[5]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[5]>>>(dI01, d_odata_dI01, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dI02, 0, arrsz, dv_streams[6]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[6]>>>(dI02, d_odata_dI02, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dI10, 0, arrsz, dv_streams[7]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[7]>>>(dI10, d_odata_dI10, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dI11, 0, arrsz, dv_streams[8]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[8]>>>(dI11, d_odata_dI11, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dI12, 0, arrsz, dv_streams[9]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[9]>>>(dI12, d_odata_dI12, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dI20, 0, arrsz, dv_streams[10]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[10]>>>(dI20, d_odata_dI20, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dI21, 0, arrsz, dv_streams[11]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[11]>>>(dI21, d_odata_dI21, size);

	gpuErrchk(cudaMemsetAsync(d_odata_dI22, 0, arrsz, dv_streams[12]));
	device_reduce_block_atomic_kernel<<< dimGrid, dimBlock, 0, dv_streams[12]>>>(dI22, d_odata_dI22, size);
	checkErrorAfterKernelLaunch("device_reduce_block_atomic_kernel");

	/* Synchronize streams before continuing with default stream */
//	for (int f=0; f<13; f++)
//		cudaStreamSynchronize(dv_streams[f]);

	/* Copy and assign */
	set_dv_dcom_di_krnl<<<1,1>>>(d_odata_dv, d_odata_dcom0, d_odata_dcom1,
			d_odata_dcom2, d_odata_dI00, d_odata_dI01, d_odata_dI02, d_odata_dI10,
			d_odata_dI11, d_odata_dI12, d_odata_dI20, d_odata_dI21, d_odata_dI22,
			dmod, c);

	/* Free up the temporary arrays */
	cudaFree(d_odata_dv);
	cudaFree(d_odata_dcom0);	cudaFree(d_odata_dcom1);	cudaFree(d_odata_dcom2);
	cudaFree(d_odata_dI00);		cudaFree(d_odata_dI01);		cudaFree(d_odata_dI02);
	cudaFree(d_odata_dI10);		cudaFree(d_odata_dI11);		cudaFree(d_odata_dI12);
	cudaFree(d_odata_dI20);		cudaFree(d_odata_dI21);		cudaFree(d_odata_dI22);

}

__host__ void sum_o2m2om_gpu(struct dat_t *ddat, double *o2, double *m2, double *om,
		int nframes, int size, int set, cudaStream_t *sb_stream) {

	/* Function sums up values o2, m2, and om for the chi2 calculation for
	 * Delay-Doppler data in double precision. */

	int f=0, maxThreads = maxThreadsPerBlock;
	int maxBlocks = 2048;
	int numBlocks = 0;
	int numThreads = 0;
	double *d_odata0, *d_odata1, *d_odata2;
	float2 xblock_ythread;
	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock;
	BLK.x = floor((THD.x - 1 + size)/THD.x);

	/* Find number of blocks & threads needed for reduction call */
	/* NOTE: This assumes all frames are the same size!      */
	xblock_ythread = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
	numBlocks = xblock_ythread.x;
	numThreads = xblock_ythread.y;
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	size_t arrsz = sizeof(double) * numBlocks;

	/* Allocate memory for d_odata and set to zero */
	gpuErrchk(cudaMalloc((void**)&d_odata0, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata1, arrsz));
	gpuErrchk(cudaMalloc((void**)&d_odata2, arrsz));
	cudaMemsetAsync(d_odata0, 0, arrsz, sb_stream[0]);
	cudaMemsetAsync(d_odata1, 0, arrsz, sb_stream[1]);
	cudaMemsetAsync(d_odata2, 0, arrsz, sb_stream[2]);

	/* Call reduction  */
	for (f=0; f<nframes; f++) {
		/* First, launch kernels to calculate o2, m2, om for this frame */
		device_reduce_block_atomic_kernel_o2<<< dimGrid,dimBlock,0,sb_stream[0]>>>
				(ddat, d_odata0, size, f, set);

		device_reduce_block_atomic_kernel_m2<<< dimGrid,dimBlock,0,sb_stream[1]>>>
				(ddat, d_odata1, size, f, set);

		device_reduce_block_atomic_kernel_om<<< dimGrid,dimBlock,0,sb_stream[2]>>>
				(ddat, d_odata2, size, f, set);

		/* Now transfer the calculated sums to variables o2, m2, and om of
		 * this frame, then set the sum arrays back to zero */
		set_o2m2om_values_krnl<<<1,1>>>(d_odata0, d_odata1,
				d_odata2, o2, om, m2, f);
		
		cudaMemsetAsync(d_odata0, 0, arrsz, sb_stream[0]);
		cudaMemsetAsync(d_odata1, 0, arrsz, sb_stream[1]);
		cudaMemsetAsync(d_odata2, 0, arrsz, sb_stream[2]);
	}
	checkErrorAfterKernelLaunch("device_reduce_block_atomic_krnl_o2/m2/om");

	cudaFree(d_odata0);
	cudaFree(d_odata1);
	cudaFree(d_odata2);
}

