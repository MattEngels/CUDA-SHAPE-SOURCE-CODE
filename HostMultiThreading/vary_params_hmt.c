
#include "../shape/head.h"

void *vary_params_hmt_sub(void *ptr);

typedef struct vp_hmt_thread_t
{
	int thread_no;
	struct par_t *parameter;
    struct mod_t *model;
    struct dat_t *data;
    double sum_deldop_zmax;
    double sum_rad_xsec;
    double sum_opt_brightness;
    double sum_cos_subradarlat;
} vphmt_data;

void vary_params_hmt(struct par_t *par, struct mod_t *mod, struct dat_t *dat,
		double *deldop_zmax, double *rad_xsec, double *opt_brightness, double
		*cos_subradarlat, pthread_t *hmt_thread)
{
	double sum_deldop_zmax = 0.0;
	double sum_rad_xsec = 0.0;
	double sum_opt_brightness = 0.0;
	double sum_cos_subradarlat = 0.0;

	/* Create the data structures for the different host threads, then load them */
	vphmt_data data[HMT_threads];
	for (int i=0; i<HMT_threads; i++) {
		data[i].thread_no = i;
		data[i].parameter = par;
		data[i].model = mod;
		data[i].data = dat;
		data[i].sum_deldop_zmax = 0.0;
		data[i].sum_rad_xsec= 0.0;
		data[i].sum_opt_brightness = 0.0;
		data[i].sum_cos_subradarlat = 0.0;
	}

	/* Now launch (HMT_threads-1) pthreads, then use the current thread to
	 * launch a vary_params_hmt_sub function on its own, then return to waiting
	 * on the other threads to complete	 */
	for (int i=0; i<(HMT_threads-1); i++)
		pthread_create(&hmt_thread[i], NULL, vary_params_hmt_sub,(void*)&data[i]);

	vary_params_hmt_sub(&data[HMT_threads-1]);

	for (int i=0; i<(HMT_threads-1); i++)
		pthread_join(hmt_thread[i], NULL);

	/* Finish the zmax, xsec, brightness, subradarlat calculations */
	for (int i=0; i<HMT_threads; i++) {
		sum_deldop_zmax += data[i].sum_deldop_zmax;
		sum_rad_xsec += data[i].sum_rad_xsec;
		sum_opt_brightness += data[i].sum_opt_brightness;
		sum_cos_subradarlat += data[i].sum_cos_subradarlat;
	}


		if (dat->sum_deldop_zmax_weights > 0.0)
			*deldop_zmax = sum_deldop_zmax / dat->sum_deldop_zmax_weights;
		else
			*deldop_zmax = 0.0;
		if (dat->sum_rad_xsec_weights > 0.0)
			*rad_xsec = sum_rad_xsec / dat->sum_rad_xsec_weights;
		else
			*rad_xsec = 0.0;
		if (dat->sum_opt_brightness_weights > 0.0)
			*opt_brightness = sum_opt_brightness / dat->sum_opt_brightness_weights;
		else
			*opt_brightness = 0.0;
		if (dat->sum_cos_subradarlat_weights > 0.0)
			*cos_subradarlat = sum_cos_subradarlat / dat->sum_cos_subradarlat_weights;
		else
			*cos_subradarlat = 0.0;
}

void *vary_params_hmt_sub(void *ptr) {

	vphmt_data *data;
	data = (vphmt_data *) ptr;

	double orbit_offset[3] = {0.0, 0.0, 0.0};
	int c, f, s, i, j, k, x, y, compute_xsec, compute_brightness, compute_zmax,
	compute_cosdelta, n, ncalc, b;
	double sum_deldop_zmax, sum_rad_xsec, sum_opt_brightness, sum_cos_subradarlat, weight,
	zmax, cross_section, oa[3][3], to_earth[3], cos_delta, intensityfactor;
	struct deldop_t *deldop;
	struct doppler_t *doppler;
	struct lghtcrv_t *lghtcrv;
	struct pos_t *pos;

	/* Process each dataset in turn  */
	for (s=0; s<data->data->nsets; s++) {
		if (data->data->set[s].inputnode==data->thread_no) {
			switch (data->data->set[s].type) {
			case DELAY:

				deldop = &data->data->set[s].desc.deldop;

				for (f=0; f<deldop->nframes; f++) {
					compute_zmax = (data->parameter->vary_delcor0 != VARY_NONE
							&& deldop->delcor.a[0].state != 'c');
					compute_xsec = (data->parameter->vary_radalb != VARY_NONE
							&& deldop->frame[f].cal.state == 'c');
					if (compute_zmax || compute_xsec) {
						weight = deldop->frame[f].weight;
						pos = &deldop->frame[f].pos;
						for (i=0; i<=2; i++)
							for (j=0; j<=2; j++) {
								pos->ae[i][j] = deldop->frame[f].view[deldop->v0].ae[i][j];
								pos->oe[i][j] = deldop->frame[f].view[deldop->v0].oe[i][j];
							}
						pos->bistatic = 0;

						/* Initialize the plane-of-sky view  */
						posclr( pos);

						/* Determine which POS pixels cover the target, and get the
						 * distance toward Earth of each POS pixel   */
						for (c=0; c<data->model->shape.ncomp; c++){
							posvis(&data->model->shape.comp[c].real, orbit_offset, pos,
									(int) data->parameter->pos_smooth, 0, 0, c);
						}

						/* Zero out the fit delay-Doppler image and call pos2deldop
						 * to create the fit image by mapping power from the plane
						 * of sky to delay-Doppler space.                             */
						clrmat(deldop->frame[f].fit, 1, deldop->frame[f].ndel,
								1, deldop->frame[f].ndop);

						pos2deldop_hmt(data->parameter, &data->model->photo, 0.0, 0.0, 0.0, deldop, 0, s, f, 0);

						/*  Compute distance toward Earth of the subradar point  */

						if (compute_zmax) {
							zmax = -HUGENUMBER;
							for (x=pos->xlim[0]; x<=pos->xlim[1]; x++)
								for (y=pos->ylim[0]; y<=pos->ylim[1]; y++)
									if (pos->cose[x][y] > 0.0)
										zmax = MAX( zmax, pos->z[x][y]);
							data->sum_deldop_zmax += zmax*weight;
						}

						/* Compute cross section  */
						if (compute_xsec) {
							cross_section = deldop->frame[f].overflow_xsec;
							for (i=1; i<=deldop->frame[f].ndel; i++)
								for (j=1; j<=deldop->frame[f].ndop; j++)
									cross_section += deldop->frame[f].fit[i][j];
							cross_section *= deldop->frame[f].cal.val;
							data->sum_rad_xsec += cross_section*weight;
						}
					}
					compute_cosdelta = (data->parameter->vary_dopscale != VARY_NONE
							&& deldop->dopscale.state != 'c');
					if (compute_cosdelta) {

						/*        oa = matrix to transform body-fixed to observer coordinates  */
						/*  to_earth = normalized target-to-Earth vector in body-fixed coords  */
						mtrnsps( oa, deldop->frame[f].view[deldop->v0].ae);
						mmmul( oa, deldop->frame[f].view[deldop->v0].oe, oa);
						for (j=0; j<=2; j++)
							to_earth[j] = oa[2][j];
						cos_delta = sqrt(to_earth[0]*to_earth[0] + to_earth[1]*to_earth[1]);
						weight = deldop->frame[f].weight;
						sum_cos_subradarlat += cos_delta*weight;
					}
				}
				break;
			case DOPPLER:

				doppler = &data->data->set[s].desc.doppler;
				for (f=0; f<doppler->nframes; f++) {

					compute_xsec = (data->parameter->vary_radalb != VARY_NONE
							&& doppler->frame[f].cal.state == 'c');

					if (compute_xsec) {
						weight = doppler->frame[f].weight;
						pos = &doppler->frame[f].pos;
						for (i=0; i<=2; i++)
							for (j=0; j<=2; j++) {
								pos->ae[i][j] = doppler->frame[f].view[doppler->v0].ae[i][j];
								pos->oe[i][j] = doppler->frame[f].view[doppler->v0].oe[i][j];
							}
						pos->bistatic = 0;

						/* Initialize the plane-of-sky view  */
						posclr( pos);

						/* Determine which POS pixels cover the target  */
						for (c=0; c<data->model->shape.ncomp; c++){
							posvis( &data->model->shape.comp[c].real, orbit_offset, pos,
									(int) data->parameter->pos_smooth, 0, 0, c);
						}
						/* Zero out the fit Doppler spectrum, then call pos2doppler to create the fit
						 * spectrum by mapping power from the plane of the sky to Doppler space.      */
						clrvect( doppler->frame[f].fit, 1, doppler->frame[f].ndop);

						pos2doppler_hmt(data->parameter, &data->model->photo, 0.0,
								0.0, 0.0, doppler, 0, s, f, 0);

						/* Compute cross section  */
						cross_section = doppler->frame[f].overflow_xsec;
						for (j=1; j<=doppler->frame[f].ndop; j++)
							cross_section += doppler->frame[f].fit[j];

						cross_section *= doppler->frame[f].cal.val;
						data->sum_rad_xsec += cross_section*weight;
					}
					compute_cosdelta = (data->parameter->vary_dopscale != VARY_NONE	&&
							doppler->dopscale.state != 'c');
					if (compute_cosdelta) {

						/* oa = matrix to transform body-fixed to observer coordinates  */
						/* to_earth = normalized target-to-Earth vector in body-fixed coords  */
						mtrnsps( oa, doppler->frame[f].view[doppler->v0].ae);
						mmmul( oa, doppler->frame[f].view[doppler->v0].oe, oa);
						for (j=0; j<=2; j++)
							to_earth[j] = oa[2][j];
						cos_delta = sqrt(to_earth[0]*to_earth[0] + to_earth[1]*to_earth[1]);
						weight = doppler->frame[f].weight;
						data->sum_cos_subradarlat += cos_delta*weight;
					}
				}
				break;
			case POS:
				break;
			case LGHTCRV:
				lghtcrv = &data->data->set[s].desc.lghtcrv;
				compute_brightness = (data->parameter->vary_optalb != VARY_NONE
						&& lghtcrv->cal.state == 'c');
				if (compute_brightness) {
					n = lghtcrv->n;
					ncalc = lghtcrv->ncalc;
					weight = lghtcrv->weight;
					for (i=1; i<=ncalc; i++) {
						pos = &lghtcrv->rend[i].pos;
						for (j=0; j<=2; j++)
							for (k=0; k<=2; k++) {
								pos->ae[j][k] = lghtcrv->rend[i].ae[j][k];
								pos->oe[j][k] = lghtcrv->rend[i].oe[j][k];
								pos->se[j][k] = lghtcrv->rend[i].se[j][k];
							}
						pos->bistatic = 1;

						/* Initialize the plane-of-sky view */
						posclr( pos);

						/* Determine which POS pixels cover the target */
						for (c=0; c<data->model->shape.ncomp; c++)
							posvis( &data->model->shape.comp[c].real, orbit_offset, pos,
									(int) data->parameter->pos_smooth, 0, 0, c);

						/* Now view the model from the source (sun) and get the facet
						 * number and distance toward the source of each pixel in this
						 * projected view; use this information to determine which POS
						 * pixels are shadowed */
						if (pos->bistatic) {
							for (c=0; c<data->model->shape.ncomp; c++)
								posvis(&data->model->shape.comp[c].real, orbit_offset, pos, 0, 1, 0, c);

							/* Identify and mask out shadowed POS pixels */
							posmask( pos, data->parameter->mask_tol);
						}

						/* Compute the model brightness for this model lightcurve
						 * point */
						intensityfactor = pow(pos->km_per_pixel/AU, 2.0);
						lghtcrv->y[i] = apply_photo(data->model, lghtcrv->ioptlaw,
								lghtcrv->solar_phase[i], intensityfactor, pos, 0, s, i);
					}

					/* Now that we have calculated the model lightcurve brightnesses
					 * y at each of the epochs x, we use cubic spline interpolation
					 * (Numerical Recipes routines spline and splint) to get model
					 * lightcurve brightness fit[i] at each OBSERVATION epoch t[i],
					 * with i=1,2,...,n.  This will allow us (in routine chi2) to
					 * compare model to data (fit[i] to obs[i]) to get chi squared.
					 * Note that vector y2 contains the second derivatives of the
					 * interpolating function at the calculation epochs x. */
					spline( lghtcrv->x, lghtcrv->y, ncalc, 2.0e30, 2.0e30, lghtcrv->y2);
					for (i=1; i<=n; i++) {
						splint( lghtcrv->x, lghtcrv->y, lghtcrv->y2, ncalc,
								lghtcrv->t[i][lghtcrv->v0], &lghtcrv->fit[i]);
						data->sum_opt_brightness += lghtcrv->fit[i]*weight;
					}
				}
				break;
			default:
				bailout("vary_params_hmt.c: can't handle this dataset type yet\n");
			}
		}
	}
	return(0);
}

