/***************************************************************************
                                                           show_deldoplim.c

For each frame of each delay-Doppler or Doppler dataset, display the
region which, according to the model, contains nonzero power.  Print
a warning whenever a region extends beyond the data limits; such
data may have to be redone with wider vignetting.

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

#include "head.h"

void show_deldoplim( struct dat_t *dat)
{
	int ndel, ndop, idellim[2], idoplim[2], s, f, i, header_displayed;
	header_displayed = 0;

	for (s=0; s<dat->nsets; s++) {

		if (dat->set[s].type == DELAY || dat->set[s].type == DOPPLER) {

			if (!header_displayed) {
				printf("#\n");
				printf("# model delay-Doppler regions (1-based) with nonzero power:\n");
				fflush(stdout);
				header_displayed = 1;
			}

			if (dat->set[s].type == DELAY) {
				for (f=0; f<dat->set[s].desc.deldop.nframes; f++) {
					ndel = dat->set[s].desc.deldop.frame[f].ndel;
					ndop = dat->set[s].desc.deldop.frame[f].ndop;

					/* Get the delay and Doppler limits, receiving them from the appro-
					 * priate branch node if the root node doesn't handle this dataset */
					for (i=0; i<=1; i++) {
						idellim[i] = dat->set[s].desc.deldop.frame[f].idellim[i];
						idoplim[i] = dat->set[s].desc.deldop.frame[f].idoplim[i];
					}

					/*  Display the limits for this frame  */
					printf("#         Set %2d frame %2d:  rows %2d to %2d , cols %2d to %2d",
							s, f, idellim[0], idellim[1], idoplim[0], idoplim[1]);
					if (idellim[1] < 1 || idellim[0] > ndel
							|| idoplim[1] < 1 || idoplim[0] > ndop)
						printf("  (MODEL ENTIRELY OUTSIDE FRAME)");
					else if (idellim[0] < 1 || idellim[1] > ndel
							|| idoplim[0] < 1 || idoplim[1] > ndop)
						printf("  (VIGNETTING TOO TIGHT)");
					printf("\n");
					fflush(stdout);
				}

			} else {
				for (f=0; f<dat->set[s].desc.doppler.nframes; f++) {
					ndop = dat->set[s].desc.doppler.frame[f].ndop;

					/*  Get the Doppler limits                */
					for (i=0; i<=1; i++)
						idoplim[i] = dat->set[s].desc.doppler.frame[f].idoplim[i];

					/*  Display the limits for this frame  */
					printf("#         Set %2d frame %2d:  bins %2d to %2d",
							s, f, idoplim[0], idoplim[1]);
					if (idoplim[1] < 1 || idoplim[0] > ndop)
						printf("  (MODEL ENTIRELY OUTSIDE FRAME)");
					else if (idoplim[0] < 1 || idoplim[1] > ndop)
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
}
