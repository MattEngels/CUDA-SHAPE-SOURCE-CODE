extern "C" {
#include "../shape/head.h"
}
__device__ void dev_euler2mat( double m[3][3], double phi, double theta, double psi)
{
	double ctheta, stheta, cphi, sphi, cpsi, spsi;

	ctheta = cos(theta);
	stheta = sin(theta);
	cphi = cos(phi);
	sphi = sin(phi);
	cpsi = cos(psi);
	spsi = sin(psi);

	m[0][0] = cpsi*cphi-ctheta*sphi*spsi;
	m[1][0] = -spsi*cphi-ctheta*sphi*cpsi;
	m[2][0] = stheta*sphi;

	m[0][1] = cpsi*sphi+ctheta*cphi*spsi;
	m[1][1] = -spsi*sphi+ctheta*cphi*cpsi;
	m[2][1] = -stheta*cphi;

	m[0][2] = spsi*stheta;
	m[1][2] = cpsi*stheta;
	m[2][2] = ctheta;
}

__global__ void euler2mat_realize_mod_krnl(struct mod_t *dmod)
{
	/* Single threaded kernel */
	double ctheta, stheta, cphi, sphi, cpsi, spsi;
	double phi = dmod->shape.comp[0].rot[0].val;
	double theta 	= dmod->shape.comp[0].rot[1].val;
	double psi 		= dmod->shape.comp[0].rot[2].val;

	if (threadIdx.x == 0) {
		ctheta = cos(theta);

		stheta = sin(theta);
		cphi = cos(phi);
		sphi = sin(phi);
		cpsi = cos(psi);
		spsi = sin(psi);

		dmod->shape.comp[0].m[0][0] = 0;
		dmod->shape.comp[0].m[0][0] = cpsi*cphi-ctheta*sphi*spsi;
		dmod->shape.comp[0].m[1][0] = -spsi*cphi-ctheta*sphi*cpsi;
		dmod->shape.comp[0].m[2][0] = stheta*sphi;

		dmod->shape.comp[0].m[0][1] = cpsi*sphi+ctheta*cphi*spsi;
		dmod->shape.comp[0].m[1][1] = -spsi*sphi+ctheta*cphi*cpsi;
		dmod->shape.comp[0].m[2][1] = -stheta*cphi;

		dmod->shape.comp[0].m[0][2] = spsi*stheta;
		dmod->shape.comp[0].m[1][2] = cpsi*stheta;
		dmod->shape.comp[0].m[2][2] = ctheta;
	}
}
