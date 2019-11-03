/***************************************************************************

                                                               read_ephem.c

Create an asteroid ephemeris structure from a standalone ephemeris file.
This file can be in the Horizons format used in shape's obs files, or
else it can be a file used for taking data at Arecibo or Goldstone.  The
"pa_ephformat" parameter signals which format to use, with Horizons format
being the default.

Modified 2012 February 1 by CM:
    Change name of "getline" routine to "readline"

Written 2003 April 23 by CM, for use with the "moments" action
    -- specifically, for finding the epoch at which the long
       principal axis lies in the plane of the sky.
***************************************************************************/

#include "head.h"

void read_ephem( struct par_t *par, struct ephem_t *ephem)
{

  FILE *ephP;
  char decsign;
  char ephLine[MAXLEN];
  int  neph, year, mon, day, hour, min, sec, rahour, ramin, decdeg, decmin, i;
  double elev, azim, rasec, decsec, delay, doppler, radeg_horizons, decdeg_horizons;

  /* Open the ephemeris file, read it to see how many points it contains
   * (ignoring blank lines and commented lines), then close it.           */
  FOPEN(ephP, par->pa_ephfile, "r");
  neph = 0;
  readline(ephP, ephLine, MAXLEN);
  while (!feof(ephP)) {
    if (!allwhite(ephLine) && ephLine[0] != '#')
      neph++;
    readline(ephP, ephLine, MAXLEN);
  }
  fclose(ephP);

  /* Allocate storage for the ephemeris structure, open the file for 2nd time,
   * and read in the ephemeris data. Adjust input format for which kind of
   * ephemeris file this is, datataking vs. Horizons.     */
  ephem->n = neph;
  ephem->pnt = (struct ephpnt_t *) calloc( neph, sizeof( struct ephpnt_t));

  FOPEN(ephP, par->pa_ephfile, "r");
  i = 0;
  while (i < neph) {
    readline(ephP, ephLine, MAXLEN);
    if (!allwhite(ephLine) && ephLine[0] != '#') {
      if (par->pa_ephformat == DATATAKING) {

          /* Read the "receive" section of a datataking ephemeris entry  */
          sscanf(ephLine,
                 "%4d-%2d-%2d %2d:%2d:%2d %lf %lf %2d%2d%lf %c%2d%2d%lf %lf %lf",
                 &year, &mon, &day, &hour, &min, &sec, &elev, &azim, &rahour,
                 &ramin, &rasec, &decsign, &decdeg, &decmin, &decsec, &delay,
                 &doppler);
          ephem->pnt[i].ra = D2R*15*(rahour + ramin/60.0 + rasec/3600.0);
          ephem->pnt[i].dec = D2R*(decdeg + decmin/60.0 + decsec/3600.0);
          if (decsign == '-')
            ephem->pnt[i].dec = -ephem->pnt[i].dec;
          ephem->pnt[i].dist = delay/(2*DAYSPERAU*86400);

      } else {
          /* Read a Horizons-format ephemeris entry  */
          sscanf(ephLine, "%d %d %d %d %d %d %lf %lf %lf",
                 &year, &mon, &day, &hour, &min, &sec, &radeg_horizons,
                 &decdeg_horizons, &ephem->pnt[i].dist);
          ephem->pnt[i].ra = D2R*radeg_horizons;
          ephem->pnt[i].dec = D2R*decdeg_horizons;
      }
      cal2jd( year, mon, day, hour, min, sec, &ephem->pnt[i].t);
      i++;
    }
  }
}
