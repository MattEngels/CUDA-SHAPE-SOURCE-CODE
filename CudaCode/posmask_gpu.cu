extern "C" {
#include "../shape/head.h"
}

__global__ void posmask_init_krnl(struct pos_t **pos, double3 *so,
		double *pixels_per_km, int size) {
	/* nfrm_alloc-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (f <= size) {
		dev_mtrnsps2(so, pos[f]->oe, f);
		dev_mmmul2(so, pos[f]->se, so, f);
		pixels_per_km[f] = 1/pos[f]->km_per_pixel;
	}
}

__global__ void posmask_krnl(
		struct par_t *dpar,
		struct pos_t **pos,
		double3 *so,
		double *pixels_per_km,
		int *posn,
		int nThreads,
		int xspan,
		int f)
{
	/* multi-threaded kernel */
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, im, jm, i1, j1, i2, j2, i_sign, j_sign;
	double3 xk;
	double i0_dbl, j0_dbl, zill, t, u;
	__shared__ int n;
	__shared__ double kmpxl, tol, bignum;

	if (threadIdx.x==0) {
		n = posn[f];
		kmpxl = pos[f]->km_per_pixel;
		tol = dpar->mask_tol;
		bignum = 0.99*HUGENUMBER;
	}
	__syncthreads();
	/*  Loop through all POS pixels  */
	if (offset < nThreads) {
		i = offset % xspan - n;
		j = offset / xspan - n;

		if (pos[f]->cose[i][j] != 0.0) {     /* if there's something there */
			xk.x = i*kmpxl;     /* calculate 3D position */
			xk.y = j*kmpxl;
			xk.z = pos[f]->z[i][j];

			/* Given the observer coordinates x of of POS pixel (i,j), find
			 * which pixel (im,jm) this corresponds to in the projected view as
			 * seen from the source (sun)             */
			dev_cotrans3(&xk, so, xk, 1, f);	/* go into source coordinates */
			i0_dbl = xk.x*pixels_per_km[f];     /* unrounded (double precision) */
			j0_dbl = xk.y*pixels_per_km[f];
			im = dev_vp_iround(i0_dbl);            /* center of nearest pixel in mask */
			jm = dev_vp_iround(j0_dbl);

			/* If center of projected pixel "seen" from source (as determined
			 * by routine posvis) lies within the boundaries of the mask,
			 * projects onto model rather than onto blank space, and represents
			 * a body, component, and facet different from those seen in the
			 * POS, calculate distance from mask pixel to source and compare to
			 * distance from POS pixel to source.                             */
			if (fabs(i0_dbl) < n && fabs(j0_dbl) < n
					&& pos[f]->zill[im][jm] > -bignum
					&&(pos[f]->f[i][j]    != pos[f]->fill[im][jm]) /*   ||
							pos[f]->comp[i][j] != pos[f]->compill[im][jm] ||
							pos[f]->body[i][j] != pos[f]->bodyill[im][jm]    )*/) {

				/* Rather than using distance towards source of CENTER of mask
				 * pixel, use bilinear interpolation to get distance towards
				 * source where the line between source and POS pixel's center
				 * intersects the mask pixel.                                */
				i1 = (int) floor( i0_dbl);
				j1 = (int) floor( j0_dbl);

				if (pos[f]->zill[i1][j1]     > -bignum &&
						pos[f]->zill[i1+1][j1]   > -bignum &&
						pos[f]->zill[i1][j1+1]   > -bignum &&
						pos[f]->zill[i1+1][j1+1] > -bignum    ) {

					/* Do standard bilinear interpolation: None of the four
					 * surrounding "grid square" pixels in the mask is
					 * blank sky                           */
					t = i0_dbl - i1;
					u = j0_dbl - j1;
					zill = (1 - t)*(1 - u)*pos[f]->zill[i1][j1]
					      + t*(1 - u)*pos[f]->zill[i1+1][j1]
					      + t*u*pos[f]->zill[i1+1][j1+1]
					      + (1 - t)*u*pos[f]->zill[i1][j1+1];
				} else {
					/* The following code block is a kludge: One or more of the
					 * four surrounding "grid square" pixels in mask is blank
					 * sky, so standard bilinear interpolation won't work.  */
					zill = pos[f]->zill[im][jm];
					i_sign = (i0_dbl >= im) ? 1 : -1;
					i2 = im + i_sign;

					if (abs(i2) <= n && pos[f]->zill[i2][jm] > -bignum) {
						zill += fabs(i0_dbl - im)  *
						(pos[f]->zill[i2][jm] - pos[f]->zill[im][jm]);
					} else {
						i2 = im - i_sign;
						if (abs(i2) <= n && pos[f]->zill[i2][jm] > -bignum)
							zill -= fabs(i0_dbl - im)
							* (pos[f]->zill[i2][jm] - pos[f]->zill[im][jm]);
					}

					j_sign = (j0_dbl >= jm) ? 1 : -1;
					j2 = jm + j_sign;

					if (abs(j2) <= n && pos[f]->zill[im][j2] > -bignum) {
						zill += fabs(j0_dbl - jm)
                        	  * (pos[f]->zill[im][j2] - pos[f]->zill[im][jm]);
					} else {
						j2 = jm - j_sign;
						if (abs(j2) <= n && pos[f]->zill[im][j2] > -bignum)
							zill -= fabs(j0_dbl - jm)
							* (pos[f]->zill[im][j2] - pos[f]->zill[im][jm]);
					}
				}
				/* If interpolated point within mask pixel is at least tol km
				 * closer to source than is the center of POS pixel, the facet
				 * represented by the mask pixel is shadowing the POS pixel:
				 * represent this by setting
				 * 		cos(scattering angle) = 0.0 for the POS pixel.      */
				if (zill - xk.z > tol)
					pos[f]->cose[i][j] = 0.0;
			}
		}
	}
}
