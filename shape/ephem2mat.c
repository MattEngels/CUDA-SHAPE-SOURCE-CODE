/*****************************************************************************************
                                                                              ephem2mat.c

Given ephem_t structures for the asteroid (ast) and the sun (sunn), and a time (t), this
routine returns the matrix which transforms from ecliptic coordinates to observer
coordinates (oe), the matrix which transforms from ecliptic coordinates to solar
coordinates (se), and the orbital contribution to the spin vector (s).  The return value
is the distance (in AU) at time t.

Set flag "bistatic" to 0 if you don't need to worry about the sun.

Modified 2015 November 7 by CM:
    Bug fix: guard against incorrect interpolation of RA and computation of RA rate as the
        asteroid (or Sun) crosses 360/0 deg

Modified 2005 July 1 by CM:
    Made error messages more informative

Modified 2005 January 25 by CM:
    Eliminated unused variable

Modified 2004 February 28 by CM:
    Added "solar_phase" argument and calculation (solar phase angle)

Modified 2004 April 9 by CM:
    Added "solar_azimuth" argument and calculation
    (asteroid-to-sun direction in the plane of the sky,
     measured from north through east)
*****************************************************************************************/

#include "head.h"

double ephem2mat( struct ephem_t ast, struct ephem_t sunn, double t, 
                  double oe[3][3], double se[3][3], double s[3], 
                  double *solar_phase, double *solar_azimuth,
                  int bistatic)
{
  int i, cd_obs[6], cd_eph[6];
  double ra_ast, dec_ast, rarate_ast, decrate_ast, dt_ast, Dt_ast, delta_ra, twopi,
         dist_ast, distrate_ast, ee[3], edot[3];
  double ra_sun, dec_sun, rarate_sun, decrate_sun, dt_sun, Dt_sun, dist_sun,
         distrate_sun, ss[3];
  double theta, phi, ss_obs[3];

  /*  Find your position in the ephemeris list  */

  i = 0;
  if (ast.pnt[0].t > t) {
    jd2cal( &cd_obs[0], &cd_obs[1], &cd_obs[2], &cd_obs[3], &cd_obs[4], &cd_obs[5], t);
    jd2cal( &cd_eph[0], &cd_eph[1], &cd_eph[2], &cd_eph[3], &cd_eph[4], &cd_eph[5],
            ast.pnt[0].t);
    printf("obs epoch %13.5f (%4d-%02d-%02d %02d:%02d:%02d)"
           " < earliest asteroid ephemeris epoch %13.5f (%4d-%02d-%02d %02d:%02d:%02d)\n",
           t,            cd_obs[0], cd_obs[1], cd_obs[2], cd_obs[3], cd_obs[4], cd_obs[5],
           ast.pnt[0].t, cd_eph[0], cd_eph[1], cd_eph[2], cd_eph[3], cd_eph[4], cd_eph[5]);
    bailout("ephem2mat: out of bounds (asteroid)\n");
  }
  while (ast.pnt[i+1].t < t) {
    ++i;
    if (i == (ast.n - 1)) {
      jd2cal( &cd_obs[0], &cd_obs[1], &cd_obs[2], &cd_obs[3], &cd_obs[4], &cd_obs[5], t);
      jd2cal( &cd_eph[0], &cd_eph[1], &cd_eph[2], &cd_eph[3], &cd_eph[4], &cd_eph[5],
              ast.pnt[0].t);
      printf("obs epoch %13.5f (%4d-%02d-%02d %02d:%02d:%02d) "
             " > latest asteroid ephemeris epoch %13.5f (%4d-%02d-%02d %02d:%02d:%02d)\n",
             t,            cd_obs[0], cd_obs[1], cd_obs[2], cd_obs[3], cd_obs[4], cd_obs[5],
             ast.pnt[i].t, cd_eph[0], cd_eph[1], cd_eph[2], cd_eph[3], cd_eph[4], cd_eph[5]);
      bailout("ephem2mat: out of bounds (asteroid)\n");
    }
  }

  /*  Interpolate angles and find rates, taking care to avoid incorrect interpolation
      between pairs of points for which RA is on opposite sides of the 360/0 deg line  */

  Dt_ast = (ast.pnt[i+1].t - ast.pnt[i].t);
  dt_ast = (t - ast.pnt[i].t);
  delta_ra = ast.pnt[i+1].ra  - ast.pnt[i].ra;
  twopi = 2*PIE;
  if (delta_ra > PIE)
    rarate_ast = (ast.pnt[i+1].ra - (ast.pnt[i].ra + twopi)) / Dt_ast;
  else if (delta_ra < -PIE)
    rarate_ast = ((ast.pnt[i+1].ra + twopi) - ast.pnt[i].ra) / Dt_ast;
  else
    rarate_ast = (ast.pnt[i+1].ra - ast.pnt[i].ra) / Dt_ast;
  ra_ast = ast.pnt[i].ra + rarate_ast*dt_ast;
  ra_ast = ra_ast - twopi*floor(ra_ast/twopi);
  decrate_ast = (ast.pnt[i+1].dec - ast.pnt[i].dec) / Dt_ast;
  dec_ast = ast.pnt[i].dec + decrate_ast*dt_ast;
  distrate_ast = (ast.pnt[i+1].dist - ast.pnt[i].dist) / Dt_ast;
  dist_ast = ast.pnt[i].dist + distrate_ast*dt_ast;

  /*  Create matrix oe  */
  /*  Unit vector which points from object to ee (in equatorial coords)  */
  /*  This is the z vector of observer coordinates  */

  oe[2][0] = -cos(dec_ast)*cos(ra_ast);
  oe[2][1] = -cos(dec_ast)*sin(ra_ast);
  oe[2][2] = -sin(dec_ast);

  /*  Vector to ee from asteroid (use this for bistatic case below)  */

  for (i=0; i<=2; i++)
    ee[i] = oe[2][i]*dist_ast;

  /*  Unit vector in -RA direction, x vector of observer coordinates  */
  /*  is Earth's pole cross z vector from above  */

  oe[0][0] = -oe[2][1];
  oe[0][1] = oe[2][0];
  oe[0][2] = 0.0;
  normalize( oe[0]);

  /*  Unit vector in DEC direction  */

  cross( oe[1], oe[2], oe[0]);

  /*  Transform everything to ecliptic coordinates  */

  for (i=0; i<=2; i++)
    eq2ec( oe[i]);

  /*  Compute s, the contribution to the apparent spin vector due to
      plane-of-sky motion.  (s is in ecliptic coordinates.)

      Note that in the code block below, oe[2] is the the unit vector
      (in ecliptic coordinates) pointing from the target toward Earth,
      and edot is its time derivative (first computed in equatorial
      coordinates and then transformed to ecliptic coordinates).        */

  edot[0] = decrate_ast*sin(dec_ast)*cos(ra_ast)
            + rarate_ast*cos(dec_ast)*sin(ra_ast);
  edot[1] = decrate_ast*sin(dec_ast)*sin(ra_ast)
            - rarate_ast*cos(dec_ast)*cos(ra_ast);
  edot[2] = -decrate_ast*cos(dec_ast);
  eq2ec( edot); 
  cross( s, edot, oe[2]);

  if (!bistatic) {
    *solar_phase = 0.0;
    *solar_azimuth = 0.0;
    return dist_ast;
  }

  /*  We're in the bistatic case from here down  */
  /*  Find your position in the ephemeris list  */

  i = 0;
  if (sunn.pnt[0].t > t) {
    jd2cal( &cd_obs[0], &cd_obs[1], &cd_obs[2], &cd_obs[3], &cd_obs[4], &cd_obs[5], t);
    jd2cal( &cd_eph[0], &cd_eph[1], &cd_eph[2], &cd_eph[3], &cd_eph[4], &cd_eph[5],
            sunn.pnt[0].t);
    printf("obs epoch %13.5f (%4d-%02d-%02d %02d:%02d:%02d)"
           " < earliest solar ephemeris epoch %13.5f (%4d-%02d-%02d %02d:%02d:%02d)\n",
           t,             cd_obs[0], cd_obs[1], cd_obs[2], cd_obs[3], cd_obs[4], cd_obs[5],
           sunn.pnt[0].t, cd_eph[0], cd_eph[1], cd_eph[2], cd_eph[3], cd_eph[4], cd_eph[5]);
    bailout("ephem2mat: out of bounds (sun)\n");
  }
  while (sunn.pnt[i+1].t < t) {
    ++i;
    if (i == (sunn.n - 1)) {
      jd2cal( &cd_obs[0], &cd_obs[1], &cd_obs[2], &cd_obs[3], &cd_obs[4], &cd_obs[5], t);
      jd2cal( &cd_eph[0], &cd_eph[1], &cd_eph[2], &cd_eph[3], &cd_eph[4], &cd_eph[5],
              sunn.pnt[i].t);
      printf("obs epoch %13.5f (%4d-%02d-%02d %02d:%02d:%02d)"
             " > latest solar ephemeris epoch %13.5f (%4d-%02d-%02d %02d:%02d:%02d)\n",
             t,             cd_obs[0], cd_obs[1], cd_obs[2], cd_obs[3], cd_obs[4], cd_obs[5],
             sunn.pnt[i].t, cd_eph[0], cd_eph[1], cd_eph[2], cd_eph[3], cd_eph[4], cd_eph[5]);
      bailout("ephem2mat: out of bounds (sun)\n");
    }
  }

  /*  Interpolate angles and find rates  */

  Dt_sun = (sunn.pnt[i+1].t - sunn.pnt[i].t);
  dt_sun = (t - sunn.pnt[i].t);
  delta_ra = sunn.pnt[i+1].ra  - sunn.pnt[i].ra;
  if (delta_ra > PIE)
    rarate_sun = (sunn.pnt[i+1].ra - (sunn.pnt[i].ra + twopi)) / Dt_sun;
  else if (delta_ra < -PIE)
    rarate_sun = ((sunn.pnt[i+1].ra + twopi) - sunn.pnt[i].ra) / Dt_sun;
  else
    rarate_sun = (sunn.pnt[i+1].ra - sunn.pnt[i].ra) / Dt_sun;
  ra_sun = sunn.pnt[i].ra + rarate_sun*dt_sun;
  ra_sun = ra_sun - twopi*floor(ra_sun/twopi);
  decrate_sun = (sunn.pnt[i+1].dec - sunn.pnt[i].dec) / Dt_sun;
  dec_sun = sunn.pnt[i].dec + decrate_sun*dt_sun;
  distrate_sun = (sunn.pnt[i+1].dist - sunn.pnt[i].dist) / Dt_sun;
  dist_sun = sunn.pnt[i].dist + distrate_sun*dt_sun;

  /*  Vector from ee to sunn  */

  ss[0] = cos(dec_sun)*cos(ra_sun)*dist_sun;
  ss[1] = cos(dec_sun)*sin(ra_sun)*dist_sun;
  ss[2] = sin(dec_sun)*dist_sun;

  /*  Vector from asteroid to sunn (in equatorial coords so far)  */

  for (i=0; i<=2; i++)
    ss[i] = ss[i] + ee[i];

  /*  Transform the asteroid-to-sun unit vector to ecliptic coordinates,
      then get se, the matrix which takes us from solar coordinates
      (where we view the target from the sun) to ecliptic coordinates     */

  eq2ec( ss);
  normalize( ss);
  theta = acos( ss[2]);
  phi = atan2( ss[1], ss[0]) + PIE/2;
  euler2mat( se, phi, theta, 0.0);

  /*  Transform the asteroid-to-sun unit vector to observer coordinates,
      then compute solar phase angle (sun-asteroid-Earth angle)
      and the asteroid-to-sun direction in the plane of the sky
      (azimuth angle measured from north through east)                    */

  cotrans( ss_obs, oe, ss, 1);
  normalize( ss_obs);
  *solar_phase = acos( ss_obs[2]);
  if (ss_obs[0] == 0.0 && ss_obs[1] == 0.0) {
      *solar_azimuth = 0.0;
  } else {
      *solar_azimuth = atan2( ss_obs[1], ss_obs[0]) - PIE/2;
      if (*solar_azimuth < 0.0)
        *solar_azimuth += 2*PIE;
  }

  return dist_ast;
}
