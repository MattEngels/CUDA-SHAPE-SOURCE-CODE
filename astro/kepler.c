/*****************************************************************************************
                                                                                 kepler.c

Use Danby's method for solving Kepler's equation using universal variables:
Danby, J. M. A., Fundamentals of Celestial Mechanics (2nd ed.), Willmann-Bell: Richmond,
1988, pp. 178-180.

Danby uses two different "f" functions: one that (along with "g") takes us from initial to
final conditions (eqn. 6.7.2); and one that expresses Kepler's equation as f(s) = 0
(eqn. 6.9.28).  To avoid confusion, the latter function has been renamed h(s) here, and
its first three derivatives are now hp, hpp, and hppp rather than fp, fpp, and fppp.

Modified 2013 June 1 by CM:
    Add ETOL defined constant
    Demand that eccentricity e >= 0.0 rather than e > -1.0
    Quit with an error message for illegal values of mu, rmin, or e, rather than quietly
        returning with displacement = velocity = 0
    For the near-parabolic case, quit with an error message if the cubic equation to be
        solved has three real roots (which shouldn't be possible given valid input
        parameters)

Converted 2007 August 4 by CM from Basic to C; simplified such that the initial epoch is
    taken to be at pericenter; and the near-parabolic case added
*****************************************************************************************/

#include "basic.h"
#include "../astro/astro.h"
#include "../util/util.h"

#define SGN(x) (((x) < 0) ? -1 : (((x) > 0) ? 1 : 0))
#define ETOL 1e-6


/*  Generate the first four Stumpff functions c0(x), c1(x), c2(x),
    and c3(x) (Danby pp. 171-174).  To maintain precision, take the
    initial argument x0 and repeatedly shrink it by factors of four
    until it is sufficiently small; then compute the functions for
    this shrunken argument x; and then use recursion relations to
    generate the function values for the initial argument x0.        */

void stumpff(double x0, double *c0, double *c1, double *c2, double *c3)
{

  int n, i;
  double x, xm;

  /*  Take initial argument x0 and repeatedly shrink it by factors
      of four until the shrunken argument x is sufficiently small   */

  x = x0;
  n = 0;
  xm = 0.1;
  while (fabs(x) >= xm) {
    n++;
    x /= 4;
  }

  /*  Get the Stumpff functions for shrunken argument x  */

  *c3 = (1 - x*(1 - x*(1 - x*(1 - x*(1 - x*(1 - x/210)/156)/110)/72)/42)/20)/6;
  *c2 = (1 - x*(1 - x*(1 - x*(1 - x*(1 - x*(1 - x/182)/132)/ 90)/56)/30)/12)/2;
  *c1 = 1 - x*(*c3);
  *c0 = 1 - x*(*c2);

  /*  Use recursion relations -- such as c1(4x) = c0(x)*c1(x) -- as
      often as required to get the functions for the initial argument x0  */

  for (i=0; i<n; i++) {
    *c3 = ((*c2) + (*c0)*(*c3))/4;
    *c2 = (*c1)*(*c1)/2;
    *c1 *= *c0;
    *c0 = 2*(*c0)*(*c0) - 1;
  }
}


/*
    Solve Kepler's equation, expressed in the form h(s) = 0
    (Danby eqn. 6.9.28, where "f(s)" is rewritten here as "h(s)")

    Based on Danby's algorithm (pp. 178-180) but simplified on the
    assumption that the initial epoch is pericenter -- which means
    that Danby's "u" parameter (dot product of initial displacement
    and initial velocity) is zero.  This simplification also makes
    it easy to input eccentricity e and then compute the "alpha"
    parameter (= -2 * specific energy) rather than vice versa.

    Input:  mu    = GM  (M = total mass)
            e     = eccentricity
            rmin  = pericenter distance
            dt    = t - t0  (time interval since pericenter)

            long_asc_node  = longitude of ascending node (rad)
            inclination    = orbital inclination (rad)
            arg_pericenter = argument of pericenter (rad)

            (The preceding three Euler angles should be given in
             equatorial coordinates)

    Output: x     = displacement vector at time dt since pericenter
            v     = velocity     vector at time dt since pericenter

            (x and v are in ecliptic coordinates)

    Return 0 if successful, 1 if no convergence to a solution
*/

int kepler(double mu, double e, double rmin, double dt,
           double long_asc_node, double inclination, double arg_pericenter,
           double *displacement, double *velocity)
{

  int n_iter, sigma, n_conway;
  double twopi, dt_use, alpha, vmax, s, a, n, P, dM, dE, dF, q, r,
         param, p1, p2, s_save, x, c0, c1, c2, c3, h, hp, hpp, hppp, ds,
         f, g, fdot, gdot, m[3][3];

  /*  Quit with an error message if total mass, pericenter distance,
      or orbital eccentricity is input with an illegal value          */

  if (mu <= 0.0 || rmin <= 0.0 || e < 0.0) {
    printf("ERROR: illegal value for total mass, pericenter distance, or eccentricity\n");
    fprintf( stderr, "ERROR: kepler.c\n");
    exit(2);
  }

  /*  Compute alpha (-2 * specific energy) and vmax (speed at pericenter)  */

  twopi = 8*atan(1.0);
  dt_use = dt;
  alpha = mu*(1 - e)/rmin;
  vmax = sqrt(mu*(1 + e)/rmin);

  /*  Generate initial guess for s, where dt = r ds and s0 = 0
      (s = integral from t0 to t of dt'/r(t') [Danby eqn. 6.9.5])  */

  if (fabs(vmax*dt_use/rmin) <= 0.2) {

      /*  For small time intervals, Danby's eqn. 6.9.30 provides
          a satisfactory "universal" initial guess for s.         */

      s = dt_use/rmin;

  } else if (fabs(1 - e) < ETOL) {

      /*  Generate initial guess for s for near-parabolic motion,
          solving a cubic equation for s (Danby eqn. 6.9.38 to 6.9.40).
          In the cubic term of eqn. 6.9.38 we substitute for alpha
          (see expression above) to obtain mu - alpha*rmin = mu*e.       */

      q = 2*rmin/(mu*e);
      r = 3*dt_use/(mu*e);
      param = q*q*q + r*r;
      if (param <= 0.0) {

        /*  Three real roots to cubic equation -- but this means q <= 0.0,
            which shouldn't be possible for valid input parameters, so we quit  */
            
        printf("ERROR: three real roots to cubic equation, illegal input parameters\n");
        fprintf( stderr, "ERROR: kepler.c\n");
        exit(3);
      }

      /*  One real root to cubic equation  */

      param = sqrt(param);
      p1 = pow(r + param, 1.0/3);
      if (r >= param)
        p2 = pow(r - param, 1.0/3);
      else
        p2 = -pow(param - r, 1.0/3);
      s = p1 + p2;

  } else if (e < 1.0) {

      /*  Generate initial guess for s for elliptic motion  */

      a = rmin/(1 - e);              /* semimajor axis a */
      n = sqrt(mu/(a*a*a));          /* mean motion n */
      P = twopi/n;                   /* orbital period P */
      dt_use -= P*floor(dt_use/P);   /* shift time interval to range [0,P) */

      /*  Want the sign of sin M (M = mean anomaly); according
          to Danby eqn. 6.8.10, this is equal to "sigma" below.
          Use this to get the initial guess for dE = E - E0,
          where E = eccentric anomaly (eqn. 6.8.9 with k = 0.85).
          Since the initial epoch is pericenter, M0 = E0 = 0.0.    */

      dM = n*dt_use;             /* M - M0 */
      sigma = SGN(e*sin(dM));    /* sign of sin M */
      dE = dM + sigma*0.85*e;    /* E - E0 */

      /*  For elliptical orbits s is directly proportional
          to E - E0 (combine Danby eqn. 6.3.8 and 6.3.9)    */

      s = dE/sqrt(alpha);

  } else {

      /*  Generate initial guess for s for hyperbolic motion
          (see Danby eqn. 6.9.35 and 6.9.36)                  */

      a = rmin/(1 - e);         /* a < 0 here */
      n = sqrt(-mu/(a*a*a));    /* mean motion n */
      dM = n*dt_use;            /* M - M0  (M = mean anomaly) */

      /*  Get the initial guess for dF = F - F0 (Danby eqn. 6.9.35
          with k = 1.8), where F = hyperbolic eccentric anomaly.
          Since the initial epoch is pericenter, M0 = F0 = 0.0.     */

      if (dM >= 0)
        dF = log(2*dM/e + 1.8);
      else
        dF = -log(-2*dM/e + 1.8);

      /*  For hyperbolic orbits s is directly proportional
          to F - F0 (Danby eqn. 6.9.36)                     */

      s = dF/sqrt(-alpha);

  }

  /*  Store initial guess for s for possible use later
      in Laguerre's method, in case Newton's method fails  */

  s_save = s;

  /*  Use Newton's method to solve Kepler's equation, written
      in the form h(s) = 0.  hp = hprime = h'(s) = dh/ds, and
      similarly hpp = h''(s) and hppp = h'''(s).               */

  n_iter = 0;

  do {

      /*  Compute the first four Stumpff functions, then
          multiply each one by the corresponding power of s  */

      x = s*s*alpha;                   /* Danby eqn. 6.9.21 */
      stumpff(x, &c0, &c1, &c2, &c3);
      c1 *= s;
      c2 *= s*s;
      c3 *= s*s*s;

      /*  Compute h(s) (where Kepler's equation reads h(s) = 0)
          and its first three derivatives (Danby eqn. 6.9.28-29)  */

      h = rmin*c1 + mu*c3 - dt_use;
      hp = rmin*c0 + mu*c2;
      hpp = mu*e*c1;
      hppp = mu*e*c0;

      /*  Improve our value of s.
          The next three lines take us from quadratic convergence
          (Newton's method) to cubic convergence (Halley's method)
          to quartic convergence: see Danby eqn. 6.6.3 to 6.6.7.    */

      ds = -h/hp;
      ds = -h/(hp + ds*hpp/2);
      ds = -h/(hp + ds*hpp/2 + ds*ds*hppp/6);
      s += ds;

      /*  Consider increasing the tolerance below (1e-12) for cases where
          hp is the sum of very large numbers that almost perfectly cancel,
          resulting in low precision for ds and an inability to converge     */

      n_iter++;

  } while (fabs(ds) >= 1e-12 && n_iter < 6);

  /*  If Newton's method failed, try Laguerre's method (Danby pp. 158-160)  */

  if (fabs(ds) >= 1e-12) {

    s = s_save;
    n_conway = 5;
    n_iter = 0;

    do {

        /*  Compute the first four Stumpff functions, then
            multiply each one by the corresponding power of s  */

        x = s*s*alpha;                   /* Danby eqn. 6.9.21 */
        stumpff(x, &c0, &c1, &c2, &c3);
        c1 *= s;
        c2 *= s*s;
        c3 *= s*s*s;

        /*  Compute h(s) (where Kepler's equation reads h(s) = 0)
            and its first two derivatives (Danby eqn. 6.9.28-29)   */

        h = rmin*c1 + mu*c3 - dt_use;
        hp = rmin*c0 + mu*c2;
        hpp = mu*e*c1;

        /*  Use Laguerre-Conway method to improve
            our value of s (Danby eqn. 6.6.26)     */

        ds = -n_conway*h
             / (hp + SGN(hp)*sqrt(fabs( (n_conway - 1)*(n_conway - 1)*hp*hp
                                          - (n_conway - 1)*n_conway*h*hpp      )));
        s += ds;

        /*  Consider increasing the tolerance below (1e-12) for cases where
            hp is the sum of very large numbers that almost perfectly cancel,
            resulting in low precision for ds and an inability to converge     */

        n_iter++;

    } while (fabs(ds) >= 1e-12 && n_iter < 20);
  }

  /*  Get the f and g functions and their time derivatives
      (Danby eqn. 6.9.27, with r = hp substituted from 6.9.29)  */

  f = 1 - (mu/rmin)*c2;
  g = dt_use - mu*c3;
  fdot = -(mu/(hp*rmin))*c1;
  gdot = 1 - (mu/hp)*c2;

  /*  Get the final displacement and velocity vectors in orbit-plane
      coordinates (Danby eqn. 6.7.2), given that the initial
      displacement and velocity are (rmin, 0, 0) and (0, vmax, 0)     */

  displacement[0] = rmin*f;
  displacement[1] = vmax*g;
  displacement[2] = 0.0;
  velocity[0] = rmin*fdot;
  velocity[1] = vmax*gdot;
  velocity[2] = 0.0;

  /*  Compute transformation matrix m, defined by Euler angles
      long_asc_node, inclination, and arg_pericenter, that takes
      us from equatorial coordinates to orbit-plane coordinates   */

  euler2mat( m, long_asc_node, inclination, arg_pericenter);

  /*  Use matrix m to convert final displacement and velocity
      from orbit-plane coordinates to equatorial coordinates
      (-1 argument = "backwards" transformation)               */

  cotrans( displacement, m, displacement, -1);
  cotrans( velocity, m, velocity, -1);

  /*  Transform final displacement and velocity from
      equatorial coordinates to ecliptic coordinates  */

  eq2ec( displacement);
  eq2ec( velocity);

  if (fabs(ds) < 1e-12)
    return 0;
  else
    return 1;   /* both Newton's and Laguerre's methods failed */
}

#undef SGN
#undef ETOL
