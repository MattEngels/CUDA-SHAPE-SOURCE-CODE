/*****************************************************************************************
                                                                               inteuler.c

This routine evolves a spin state from the initial spin epoch to a desired final time by
integrating Euler's equations and, if specified, applying one or more spin impulses at
various epochs along the way.  Input includes a structure of type spin_t giving initial
conditions for the model's spin state (in ecliptic coordinates); an n-length vector of
times t to integrate to; the spin impulses to be applied at those times, impulse[n][3],
a matrix listing instantaneous increments to the spin vector components as expressed in
body-fixed coordinates; and the "pa" and "method" flags that are described below.  The
initial impulse (impulse[0][*]) will always be zero, and the final one will usually be
zero as well, unless by coincidence the final time t[n-1] precisely corresponds to the
epoch at which a spin impulse is to be applied.

The evolution must be done in steps if spin impulses are being applied at one or more
times between t[0] (which is always the initial spin epoch spin.t0) and final time t[n-1].
The routine takes the initial spin state and evolves it from t[0] to t[1]; then the spin
impulse at t[1] (= impulse[1][*]) is added to the evolved spin vector (expressed in
body-fixed coordinates); then this new spin state is evolved from t[1] to t[2]; then the
second spin impulse is added to the spin vector output by that integration; and so on
until we reach the desired epoch t[n-1].  If t[n-1] < t[0] then we are integrating
backwards in time and the spin impulses are negated by the realize_spin routine,
so that inteuler effectively subtracts them from the spin vector rather than adding them.

The "pa" flag is true if the model is in a principal-axis rotation state, in which case
the spin state is evolved using simple arithmetic rather than by integrating Euler's
equations.  The "method" flag selects the integration method for non-principal-axis (NPA)
rotation:

    method = BRUTE     : brute-force numerical integration from t[0] to t[n-1]
           = PERIODIC  : numerical integration exploiting periodicity of the spin vector

Output is the spin vector w[3] (in body-fixed coordinates) at time t[n] and also the
matrix m[3][3] that transforms from ecliptic to body-fixed coordinates at time t[n-1].

Modified 2014 August 22 by SN:
    Included libration along the z direction in the spin

Modified 2011 August 23 by CM:
    When integrating Euler's equations for an NPA rotator, pass the "int_abstol"
        parameter to the Numerical Recipes in C "odeint" routine rather than using
        a hard-wired tolerance
    When integrating Euler's equations for an NPA rotator, scale the spin vector
        components by wscale, the magnitude of the initial spin vector, so that all 12
        variables being solved for (3 for w, 9 for m) are of order unity.  Until now
        the solution for m was somewhat less accurate than that for w.  (The elegant
        thing would be to use dimensionless time t' = wscale*t but I won't bother.)
    Implement spin impulses.  This entails changing the arguments and evolving the spin
        state in multiple steps.  For NPA rotators it also entails changing the dev_intnpa1
        and dev_intnpa2 routines so that they (a) don't assume that the integration starts at
        initial spin epoch spin.t0 with spin state given by spin.angle and spin.omega,
        and (b) use "w" and "m" as input parameters (initial spin vector and angular
        orientation) as well as output parameters.
    Improve comments

Modified 2008 December 12 by CM:
    Adjust dev_intnpa1 routine so that it doesn't choke on the special case
        of integrating Euler's equations over zero time interval
    Adjust comments

Modified 2007 September 20 by CM:
    Make "method" parameter unsigned char rather than int
    Adjust comments

Modified 2006 March 6 by PT:
    Instances of spin.omega[i].val replaced with spin.omega[i].val
         + spin.omegadot[i].val*(t - spin.t0) to account for linear
         change in spin rate (YORP)

Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflicts

Modified 2005 January 25 by CM:
    Took care of unused and uninitialized variables
    Removed commented-out code block at end of routine intnpa1
 ***************************************************************************/
extern "C" {
#include "../shape/head.h"
}

__device__ void dev_derivs( double x, double *y, double *dydx);
__device__ void dev_intnpa1( double t0, double t, double inertia[3], double w[3], double m[3][3], double eps);
__device__ void dev_intnpa2( double t0, double t, double inertia[3], double w[3], double m[3][3], double eps);

__device__ double I[4], wscale;

__device__ void dev_inteuler( struct spin_t spin, double t[], double impulse[][3], int n,
		double w[3], double m[3][3], unsigned char pa, unsigned char method,
		double int_abstol)
{
	int i, k;
	double psi, inertia[3];

	if (pa) {
		/* PA rotation: the x and y components of the spin vector (spin.omega),
		 * YORP acceleration (spin.omegadot), and all spin impulses are held
		 * fixed at 0.0, so we can evolve the spin state via simple kinematics
		 * rather than by solving Euler's equations  */

		psi = spin.angle[2].val + spin.omega[2].val*(t[n-1] - t[0])
            						+ 0.5*spin.omegadot[2].val*(t[n-1] - t[0])*(t[n-1] - t[0])
            						+ spin.lib_amp.val * sin(spin.lib_freq.val*(t[n-1] - t[0]) + spin.lib_phase.val);
		for (i=0; i<=1; i++)
			w[i] = spin.omega[i].val + spin.omegadot[i].val*(t[n-1] - t[0]);
		/* apply libration to z component of spin only */
		i = 2;
		w[i] = spin.omega[i].val + spin.omegadot[i].val*(t[n-1] - t[0]) + spin.lib_amp.val*spin.lib_freq.val*cos(spin.lib_freq.val*(t[n-1] - t[0]) + spin.lib_phase.val);
		for (k=0; k<n; k++) {
			psi += impulse[k][2]*(t[n-1] - t[k]);
			w[2] += impulse[k][2];
		}
		dev_euler2mat( m, spin.angle[0].val, spin.angle[1].val, psi);

	} else {
		/* NPA rotation: currently can't combine with YORP  */
		/* Create a vector containing the principal moments of inertia  */
		for (i=0; i<=2; i++)
			inertia[i] = spin.inertia[i].val;

		/* Assign global variable wscale = |initial spin vector|; this is the
		 * factor by which spin vector components are reduced to make them of
		 * order unity. It is used by the dev_intnpa1 and dev_intnpa2 routines and by the
		 * dev_derivs routine (called by the Numerical Recipes in C "odeint" routine,
		 * the driver for integrating Euler's equations).       */
		wscale = 0.0;
		for (i=0; i<=2; i++)
			wscale += spin.omega[i].val * spin.omega[i].val;
		wscale = sqrt(wscale);

		/* Initialize spin state  */
		dev_euler2mat( m, spin.angle[0].val, spin.angle[1].val, spin.angle[2].val);
		for (i=0; i<=2; i++)
			w[i] = spin.omega[i].val;

		/* Evolve the spin state from t[0] (= spin.t0) to t[n-1] one interval at
		 * a time, adding the appropriate spin impulse to the spin vector after
		 * each interval. Note that w and m are input parameters to the dev_intnpa1
		 * and dev_intnpa2 routines, which then change/evolve them and return them as
		 * output parameters.            */
		switch (method) {
		case BRUTE:
			for (k=0; k<n-1; k++) {
				dev_intnpa1( t[k], t[k+1], inertia, w, m, int_abstol);
				for (i=0; i<=2; i++)
					w[i] += impulse[k+1][i];
			}
			break;
		case PERIODIC:
			for (k=0; k<n-1; k++) {
				dev_intnpa2( t[k], t[k+1], inertia, w, m, int_abstol);
				for (i=0; i<=2; i++)
					w[i] += impulse[k+1][i];
			}
			break;
		default:
			printf("method %d\n", method);
			printf("inteuler.c: don't know that method\n");
		}
	}
}


/*  The "dev_derivs" routine is called by the Numerical Recipes in C "odeint" driver
    routine, which in turn is called by the dev_intnpa1 routine.  It evaluates the
    time derivatives of the 12 variables we're solving for (three components of
    body-fixed spin vector w, nine elements of ecliptic-to-body-fixed coordinate
    transformation matrix m) when evolving the spin state                         */
/* Update 9/27/2017 : routine now works with a zero-index */
__device__ void dev_derivs( double x, double *y, double *dydx)
{
	int i;

	/*  Vector y contains the values of the twelve variables we're solving for:      */
	/*                                                                               */
	/*  y[ 1],y[ 2],y[ 3] = rescaled spin  in body-fixed coords = w0,w1,w2 / wscale  */
	/*  y[ 4],y[ 5],y[ 6] = ecliptic x-hat in body-fixed coords = m00,m10,m20        */
	/*  y[ 7],y[ 8],y[ 9] = ecliptic y-hat in body-fixed coords = m01,m11,m21        */
	/*  y[10],y[11],y[12] = ecliptic z-hat in body-fixed coords = m02,m12,m22        */

	/*  Get time derivatives of spin vector components via Euler's equations:

                d(w_x)/dt  =  [(I_y - I_z)/I_x] * w_y * w_z
                d(w_y)/dt  =  [(I_z - I_x)/I_y] * w_z * w_x
                d(w_z)/dt  =  [(I_x - I_y)/I_z] * w_x * w_y                        */

	dydx[1] = wscale*((I[2] - I[3])/I[1])*y[2]*y[3];
	dydx[2] = wscale*((I[3] - I[1])/I[2])*y[3]*y[1];
	dydx[3] = wscale*((I[1] - I[2])/I[3])*y[1]*y[2];

	/*  Get time derivatives of ecliptic-to-body-fixed coordinate transformation
      matrix elements, which determine the orientation of the ecliptic axes as
      expressed in body-fixed coordinates (see above):

                d(ecliptic x-hat)/dt  =  -(spin vector) x (ecliptic x-hat)
                d(ecliptic y-hat)/dt  =  -(spin vector) x (ecliptic y-hat)
                d(ecliptic z-hat)/dt  =  -(spin vector) x (ecliptic z-hat)

      -- where the minus signs result from the "backwards" rotation of the
         ecliptic axes as seen in the body-fixed reference frame                   */

	for (i=0; i<=6; i+=3) {
		dydx[4+i] = wscale*(y[5+i]*y[3] - y[6+i]*y[2]);
		dydx[5+i] = wscale*(y[6+i]*y[1] - y[4+i]*y[3]);
		dydx[6+i] = wscale*(y[4+i]*y[2] - y[5+i]*y[1]);
	}
}


/*  Brute-force numerical integration of Euler's equations  */

__device__ void dev_intnpa1( double t0, double t, double inertia[3], double w[3], double m[3][3], double eps)
{
	int i, j, k, nok, nbad;
	double y[13];
	y[0] = 0.0;
	/*  If the integration timespan is zero, the "odeint" routine (called
      below) will choke, so instead of calling it we immediately return
      with the "final" values of w and m left at the initial values      */

	if (fabs(t - t0) < 1.0e-6)
		return;

	/*  Assign the principal moments of inertia to global variable (vector) I
      so that the "dev_derivs" routine (called by "odeint") can access them      */
	for (i=0; i<=2; i++)
		I[i+1] = inertia[i];

	/* Create vector y and fill it with initial values of the 12 variables we're solving
      for: three components of w, the sidereal spin vector in body-fixed coordinates,
      rescaled to be of order unity; and nine elements of m, the ecliptic-to-body-fixed
      coordinate transformation matrix that describes the model's angular orientation.
      See comments in the "dev_derivs" routine for details on the elements of y.             */
	k = 0;
	for (i=0; i<=2; i++)
		y[++k] = w[i]/wscale;
	for (j=0; j<=2; j++)
		for (i=0; i<=2; i++)
			y[++k] = m[i][j];

	/* Evolve the spin state from t0 to t by numerically solving 12 coupled first-order
      differential equations in time, three for w (Euler's equations) and nine for m;
      see comments in the "dev_derivs" routine for details on the equations, and Chapter 16
      of Numerical Recipes in C for details on the integration method                    */

	dev_odeint( y, t0, t, eps, 0.01, 0.0, &nok, &nbad, dev_derivs, dev_bsstep);

	/*  Copy the evolved values in y back to w (taking out the scaling factor) and m  */

	k = 0;
	for (i=0; i<=2; i++)
		w[i] = y[++k]*wscale;
	for (j=0; j<=2; j++)
		for (i=0; i<=2; i++)
			m[i][j] = y[++k];
}

__device__ void dev_piksrt(int n,double *arr)
{
	int i,j;
	double a;

	for (j=2;j<=n;j++) {
		a=arr[j];
		i=j-1;
		while (i > 0 && arr[i] > a) {
			arr[i+1]=arr[i];
			i--;
		}
		arr[i+1]=a;
	}
}

__device__ int dev_iround(double x)
{
	if (x < 0.0)
		return ((int)(x - 0.5));
	else
		return ((int)(x + 0.5));
}


/*  Numerical integration of Euler's equations that exploits the periodicity of spin vector w  */

__device__ void dev_intnpa2( double t0, double t, double inertia[3], double w[3], double m[3][3], double eps)
{
	int i, j, N;
	double w0[3], m0[3][3], wtmp, L[3], tmp, twoE, M2, k2, K, T, dt, Ltheta, Lphi,
	wT[3], mT[3][3], min, u[3], v[3], c, s, psiT, m1[3][3], m2[3][3];

	/* Initialize variable to avoid compilation warning  */
	j = 0;

	/* Save input values of w and m for later use  */

	for (i=0; i<=2; i++)
		w0[i] = w[i];

	for (i=0; i<=2; i++)
		for (j=0; j<=2; j++)
			m0[i][j] = m[i][j];

	/* Assign principal moments of inertia and compute angular momentum,
	 * twoE (= 2*energy) and M2 (= |angular momentum|^2)                  */
	twoE = M2 = 0.0;
	for (i=1; i<=3; i++) {
		wtmp = w[i-1];                                                      /* spin component */
		I[i] = inertia[i-1];                                                /* moment of inertia */
		L[i-1] = I[i]*wtmp;                                                 /* ang. momentum component */
		tmp = I[i]*wtmp*wtmp;
		twoE += tmp;                                                        /* 2*energy */
		M2 += I[i]*tmp;                                                     /* |angular momentum|^2 */
	}

	/* Sort the principal moments of inertia from smallest to largest  */
	dev_piksrt( 3, I);

	/* Calculate T, the period of w in the body-fixed frame (see Landau &
	 * Lifshitz, Mechanics, section 37)        */
	if (M2 < twoE*I[2]) {

		/* Landau p. 118: Rather than having two formulas for short-axis vs.
		 * long-axis modes we switch the smallest and largest principal moments
		 * for a long-axis mode          */
		tmp = I[1];
		I[1] = I[3];
		I[3] = tmp;
	}
	k2 = ((I[2] - I[1])*(twoE*I[3] - M2)) / ((I[3] - I[2])*(M2 - twoE*I[1]));
	K = dev_cel( sqrt(1 - k2), 1.0, 1.0, 1.0);
	T = 4*K*sqrt( (I[1]*I[2]*I[3]) / ((I[3] - I[2])*(M2 - twoE*I[1])) );

	/* Transform the angular momemtum vector to ecliptic coordinates,
	 * normalize it, and calculate Euler angles phi and theta          */
	dev_cotrans2( L, m0, L, -1);
	dev_normalize( L);
	Ltheta = acos( L[2]);
	Lphi = atan2( L[1], L[0]) + PIE/2;

	/*  Break integration time up as t-t0 = NT+dt  */
	N = dev_iround((t - t0)/T);
	dt = t - t0 - N*T;

	/*  Integrate motion over one T, result is pure rotation about L
	 * The spin vector output by this call to dev_intnpa1 should in principle be
	 * identical to the input value but will in fact be slightly different due
	 * to the small-but-finite tolerance for the numerical integration; thus we
	 * use a copy (wT) so as not to affect w.  We also use a copy of the
	 * transformation matrix (mT) but dev_intnpa1 will change this even in
	 * principle, and the output matrix will actually be used later on. Note
	 * that dev_intnpa1 reassigns the principal moments of inertia to the I vector,
	 * so it doesn't matter that we reordered I earlier.            */

	for (i=0; i<=2; i++)
		wT[i] = w0[i];

	for (i=0; i<=2; i++)
		for (j=0; j<=2; j++)
			mT[i][j] = m0[i][j];

	dev_intnpa1( t0, t0 + T, inertia, wT, mT, eps);

	/* Create unit vector orthogonal to L  */

	min = 1.0;
	for (i=0; i<=2; i++) {
		tmp = fabs(L[i]);
		if (tmp < min) {
			min = tmp;
			j = i;
		}
	}
	u[0] = u[1] = u[2] = 0.0;
	u[j] = 1.0;
	dev_cross( u, L, u);
	dev_normalize( u);

	/* Find how u transforms over time T  */
	dev_cotrans2( v, m0, u, 1);                /* ecliptic to body at t0   */
	dev_cotrans2( v, mT, v, -1);               /* body to ecliptic at t0+T */
	c = dev_dot( u, v);                               /* cos(psi) */
	dev_cross( u, u, v);
	s = dev_dot( L, u);                               /* sin(psi) */
	psiT = atan2( s, c);

	/* Integrate motion over one dt:
	 * accounts for everything except pure rotation about L (w and m should
	 * still have their initial values w0 and m0 prior to this call to dev_intnpa1*/
	dev_intnpa1( t0, t0 + dt, inertia, w, m, eps);

	/* Adjust the angular orientation for pure rotation about L over time NT  */
	dev_euler2mat( m1, Lphi, Ltheta, N*psiT);
	dev_euler2mat( m2, 0.0, -Ltheta, -Lphi);
	dev_mmmul( mT, m2, m1);
	dev_mmmul( m, m, mT);
}
