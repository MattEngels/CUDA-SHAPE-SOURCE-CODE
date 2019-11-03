/***************************************************************************
                                                                    ec2eq.c

Converts Cartesian vector u from ecliptic coordinates to equatorial
coordinates.  Both input and output use the J2000.0 equinox: 2000 January 1
at 12:00:00 Terrestrial Time (TT).

Modified by CM on 2007 March 18:
    Update COSEPS and SINEPS constants to reflect the updated value of the
        obliquity of the ecliptic for J2000.0 (epsilon_0 = 23d 26' 21.406")
        according to the IAU 2006 precession model (J. L. Hilton et al.,
        2006, Celestial Mechanics and Dynamical Astronomy, 94, 351-367)
***************************************************************************/

#include "basic.h"

/* cosine and sine of the obliquity of the ecliptic for J2000.0 */
#define COSEPS 0.9174821430652418
#define SINEPS 0.3977769691126060

void ec2eq( double *u)
{
  double tmp;

  tmp  = u[1]*COSEPS - u[2]*SINEPS;
  u[2] = u[2]*COSEPS + u[1]*SINEPS;
  u[1] = tmp;
}

#undef COSEPS
#undef SINEPS
