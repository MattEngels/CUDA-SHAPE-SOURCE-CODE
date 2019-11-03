/***************************************************************************
                                                           show_deldoplim.c

For each frame of each delay-Doppler or Doppler dataset, display the
region which, according to the model, contains nonzero power.  Print
a warning whenever a region extends beyond the data limits; such
data may have to be redone with wider vignetting.

Modified 2016 November 18 by ME:
	Converted to CUDA code, to run on CUDA-capable device only, with very 
	little CPU calculation.

Modified 2009 April 3 by CM:
    For MPI_Recv calls, mpi_par[0] is no longer equal to the MPI action,
        since the message tag argument already serves that purpose (as of
        2008 April 10) -- so the other mpi_par elements are renumbered

Modified 2008 April 10 by CM:
    Use message tag argument to MPI_Recv to identify the MPI action

Modified 2007 August 18 by CM:
    Rename MPI_TAG to MPI_TAG_1 to avoid name conflict with mpich headers

Modified 2006 June 18 by CM:
    Allow each delay-Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow each Doppler frame within a dataset to have different
        dimensions after vignetting

Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflicts

Modified 2005 June 25 by CM:
    Renamed "dellim" to "idellim" and "doplim" to "idoplim"

Modified 2005 February 13 by CM:
    For the "fit" action with parallel processing, revise the code (and
        the code in branch.c) so that only root calls show_deldoplim and
        the branch nodes send their (delay-)Doppler limits to root to be
        displayed by root.  This ensures that the screen output will be
        ordered by dataset.

Modified 2005 January 12 by CM:
    For the "fit" action with parallel processing, revise the code so
        that it will still work: For each dataset which is handled by a
        branch node rather than by root, root broadcasts a request for
        that branch node to run show_deldoplim for just that one dataset
        and to report back to root that the operation is complete.  This
        is necessary because each node only "knows" about a subset of
        the data, so we must have different nodes process different
        datasets -- and process them in order so that the screen display
        comes out in order.

Modified 2004 July 30 by CM:
    Add a special warning if the model power lies entirely outside the
        data frame

Modified 2004 February 20 by CM:
    Don't display the header line if there are no delay-Doppler or
        Doppler datasets

Written 2003 April 26 by CM
 ***************************************************************************/
extern "C" {
#include "../shape/head.h"
}

/* Function needs very little conversion.  Most can keep happening on the host.
 * Only copy what is needed (and has been updated) from the device.  */
__global__ void sho_ddl_get_lims_krnl(struct dat_t *ddat, int2 *idellim,
		int2 *idoplim, int *ndel, int *ndop, int s, int nframes) {
	/* nframes-threaded kernel*/
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < nframes) {
		if (ddat->set[s].type == DELAY) {
			idellim[f].x = ddat->set[s].desc.deldop.frame[f].idellim[0];
			idellim[f].y = ddat->set[s].desc.deldop.frame[f].idellim[1];
			idoplim[f].x = ddat->set[s].desc.deldop.frame[f].idoplim[0];
			idoplim[f].y = ddat->set[s].desc.deldop.frame[f].idoplim[1];
			ndel[f] = ddat->set[s].desc.deldop.frame[f].ndel;
			ndop[f] = ddat->set[s].desc.deldop.frame[f].ndop;
		}
		if (ddat->set[s].type == DOPPLER) {
			idoplim[f].x = ddat->set[s].desc.doppler.frame[f].idoplim[0];
			idoplim[f].y = ddat->set[s].desc.doppler.frame[f].idoplim[1];
			ndop[f] = ddat->set[s].desc.doppler.frame[f].ndop;
		}
	}
}
__host__ void show_deldoplim_gpu(struct dat_t *ddat,
		unsigned char *type, int nsets, int *nframes, int maxframes)
 {
 	int *ndel, *ndop, *hndel, *hndop, s, f, header_displayed;
 	header_displayed = 0;
 	int2 *idellim, *idoplim, *hidellim, *hidoplim;
 	dim3 BLK[nsets],THD;
 	THD.x = maxThreadsPerBlock;

 	/* Allocate host and device memory */
 	gpuErrchk(cudaMalloc((void**)&ndel, sizeof(int) * maxframes));
 	gpuErrchk(cudaMalloc((void**)&ndop, sizeof(int) * maxframes));
 	gpuErrchk(cudaMalloc((void**)&idoplim, sizeof(int2) * maxframes));
 	gpuErrchk(cudaMalloc((void**)&idellim, sizeof(int2) * maxframes));

 	hndel 	 = (int *) malloc(maxframes*sizeof(int));
 	hndop 	 = (int *) malloc(maxframes*sizeof(int));
 	hidellim = (int2 *) malloc(maxframes*sizeof(int2));
 	hidoplim = (int2 *) malloc(maxframes*sizeof(int2));

 	for (s=0; s<nsets; s++) {

 		if (type[s] == DELAY || type[s] == DOPPLER) {

 			if (!header_displayed) {
 				printf("#\n");
 				printf("# model delay-Doppler regions (1-based) with nonzero power:\n");
 				fflush(stdout);
 				header_displayed = 1;
 			}

 			if (type[s] == DELAY) {
 				BLK[s] = floor((THD.x - 1 + nsets) / THD.x);

 				/* Get the delay and Doppler limits*/
 				sho_ddl_get_lims_krnl<<<BLK[s],THD>>>(ddat, idellim,
 						idoplim, ndel, ndop, s, nframes[s]);
 				checkErrorAfterKernelLaunch("sho_ddl_get_lims_krnl");
 				gpuErrchk(cudaMemcpy(hidellim, idellim, sizeof(int2)*nframes[s],
 						cudaMemcpyDeviceToHost));
 				gpuErrchk(cudaMemcpy(hidoplim, idoplim, sizeof(int2)*nframes[s],
 						cudaMemcpyDeviceToHost));
 				gpuErrchk(cudaMemcpy(hndel, ndel, sizeof(int)*nframes[s],
 						cudaMemcpyDeviceToHost));
 				gpuErrchk(cudaMemcpy(hndop, ndop, sizeof(int)*nframes[s],
 						cudaMemcpyDeviceToHost));

 				for (f=0; f<nframes[s]; f++) {
 				/*  Display the limits for this frame  */
 				printf("#         Set %2d frame %2d:  rows %2d to %2d , cols %2d to %2d",
 						s, f, hidellim[f].x, hidellim[f].y, hidoplim[f].x, hidoplim[f].y);
 				if (hidellim[f].y < 1 || hidellim[f].x > hndel[f]
 						|| hidoplim[f].y < 1 || hidoplim[f].x > hndop[f])
 					printf("  (MODEL ENTIRELY OUTSIDE FRAME)");
 				else if (hidellim[f].x < 1 || hidellim[f].y > hndel[f]
 						|| hidoplim[f].x < 1 || hidoplim[f].y > hndop[f])
 					printf("  (VIGNETTING TOO TIGHT)");
 				printf("\n");
 				fflush(stdout);
 				}

 			} else {
 				BLK[s] = floor((THD.x - 1 + nsets) / THD.x);

 				/* Get the delay and Doppler limits */
 				sho_ddl_get_lims_krnl<<<BLK[s],THD>>>(ddat, idellim,
 						idoplim, ndel, ndop, s, nframes[s]);
 				checkErrorAfterKernelLaunch("sho_ddl_get_lims_krnl");
 				gpuErrchk(cudaMemcpy(hidoplim, idoplim, sizeof(int2)*nframes[s],
 						cudaMemcpyDeviceToHost));
 				gpuErrchk(cudaMemcpy(hndop, ndop, sizeof(int)*nframes[s],
 						cudaMemcpyDeviceToHost));

 				for (f=0; f<nframes[s]; f++) {
 					/*  Display the limits for this frame  */
 					printf("#         Set %2d frame %2d:  bins %2d to %2d",
 							s, f, hidoplim[f].x, hidoplim[f].y);
 					if (hidoplim[f].y < 1 || hidoplim[f].y > hndop[f])
 						printf("  (MODEL ENTIRELY OUTSIDE FRAME)");
 					else if (hidoplim[f].x < 1 || hidoplim[f].y > hndop[f])
 						printf("  (VIGNETTING TOO TIGHT)");
 					printf("\n");
 					fflush(stdout);
 				}
 			}
 		}
 	}  /* end loop over datasets */

 	if (header_displayed) {
 		printf("#\n");
 		fflush(stdout);
 	}
 	free(hndop);
 	free(hndel);
 	free(hidellim);
 	free(hidoplim);
 	cudaFree(ndop);
 	cudaFree(ndel);
 	cudaFree(idellim);
 	cudaFree(idoplim);
 }
