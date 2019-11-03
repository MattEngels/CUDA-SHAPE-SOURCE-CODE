#include "../shape/head.h"


void *calc_fits_hmt_sub(void *ptr);

typedef struct cf_hmt_thread_t
{
	int thread_no;
	struct par_t *parameter;
    struct mod_t *model;
    struct dat_t *data;
    double sum_deldop_zmax;
    double sum_rad_xsec;
    double sum_opt_brightness;
    double sum_cos_subradarlat;
} cfhmt_data;


void calc_fits_hmt( struct par_t *par, struct mod_t *mod, struct dat_t *dat, pthread_t *hmt_thread)
{
	int c, f, s, frm, i;

	/* Initialize flags that indicate the model extends beyond POS frame, that
	 * plane-of-sky fit images are too small to "contain" the target, and that
	 * model is too wide in (delay-)Doppler space to create (delay-)Doppler fit
	 * frames                                  */

	par->posbnd = 0;
	par->badposet = 0;
	par->badradar = 0;
	par->posbnd_logfactor = 0.0;
	par->badposet_logfactor = 0.0;
	par->badradar_logfactor = 0.0;

	/*  Initialize the flags that indicate whether or not each facet
      of each model component is ever visible and unshadowed from Earth  */

	for (c=0; c<mod->shape.ncomp; c++)
		for (f=0; f<mod->shape.comp[c].real.nf; f++)
			mod->shape.comp[c].real.f[f].seen = 0;

	/*  For the "write" action, announce whether or not epochs
      have been corrected for one-way light travel time,
      and create the "listpos_path" directory if necessary    */

	if (par->action == WRITE) {
		printf("#\n");
		if (par->perform_ltc) {
			printf("# Epochs listed below are times when light left the target\n");
		} else {
			printf("# Epochs listed below are times when a terrestrial observer\n");
			printf("# would receive light from the target\n");
		}
		printf("#\n");

		if (par->listpos_deldop || par->listpos_opt)
			if (!createdir(par->listpos_path)) {
				printf("Unable to create 'listpos_path' directory\n");
				bailout("calc_fits.c: program halted\n");
			}
	}

	/* Create the data structures for the different host threads, then load them */
	cfhmt_data data[HMT_threads];
	for (int i=0; i<HMT_threads; i++) {
		data[i].thread_no = i;
		data[i].parameter = par;
		data[i].model = mod;
		data[i].data = dat;		
	}
	
	/* Now launch (HMT_threads-1) pthreads, then use the current thread to
	 * launch a vary_params_hmt_sub function on its own, then return to waiting
	 * on the other threads to complete	 */
	for (int i=0; i<(HMT_threads-1); i++)
		pthread_create(&hmt_thread[i], NULL, calc_fits_hmt_sub,(void*)&data[i]);

	calc_fits_hmt_sub(&data[HMT_threads-1]);

	for (int i=0; i<(HMT_threads-1); i++)
		pthread_join(hmt_thread[i], NULL);

	/* Now finish up other stuff */
	/*  Complete the calculations of values that will be used during a fit
      to increase the objective function for models with bad properties   */

	par->posbnd_logfactor /= dat->dof;
	par->badposet_logfactor /= dat->dof_poset;
	par->badradar_logfactor /= (dat->dof_deldop + dat->dof_doppler);
	fflush(stdout);

	/*  For the "write" action with the "mark_unseen" parameter
      turned on, go back and create the colored POS images     */

	if (par->action == WRITE && par->mark_unseen) {
		for (s=0; s<dat->nsets; s++) {
			switch (dat->set[s].type) {
			case DELAY:
				for (frm=0; frm<dat->set[s].desc.deldop.nframes; frm++)
					write_pos_deldop( par, mod, &dat->set[s].desc.deldop, s, frm);
				break;
			case DOPPLER:
				for (frm=0; frm<dat->set[s].desc.doppler.nframes; frm++)
					write_pos_doppler( par, mod, &dat->set[s].desc.doppler, s, frm);
				break;
			case POS:
				for (frm=0; frm<dat->set[s].desc.poset.nframes; frm++)
					write_pos_poset( par, mod, &dat->set[s].desc.poset, s, frm);
				break;
			case LGHTCRV:
				if (par->lcrv_pos)
					for (i=1; i<=dat->set[s].desc.lghtcrv.ncalc; i++)
						write_pos_lghtcrv( par, mod, &dat->set[s].desc.lghtcrv, s, i);
				break;
			}
		}
	}
}

	
void *calc_fits_hmt_sub(void *ptr) {

	cfhmt_data *data;
	data = (cfhmt_data *) ptr;
	int s;
	
	/* Calculate the fits for each dataset in turn  */

	for (s=0; s<data->data->nsets; s++) {
		if (data->data->set[s].inputnode==data->thread_no) {
		switch (data->data->set[s].type) {
		case DELAY:
			calc_deldop(data->parameter, data->model, &data->data->set[s].desc.deldop, s);
			break;
		case DOPPLER:
			calc_doppler(data->parameter, data->model, &data->data->set[s].desc.doppler, s);
			break;
		case POS:
			calc_poset(data->parameter, data->model, &data->data->set[s].desc.poset, s);
			break;
		case LGHTCRV:
			calc_lghtcrv(data->parameter, data->model, &data->data->set[s].desc.lghtcrv, s);
			break;
		default:
			bailout("calc_fits_hmt_sub.c: can't handle this type yet\n");
		}
		}
	}
	return(0);
}
