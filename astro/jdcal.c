/*  jd2cal modified 2003 April 15 by CM to avoid getting "60" in the seconds place  */

#include "basic.h"
//#include <nr.h>
#include "../util/util.h"


void cal2jd( int yy, int mo, int dd, int hh, int mm, int ss,
             double *jd)
{
  *jd = julday( mo, dd, yy) - 0.5;
  *jd += ((hh*60 + mm)*60 + ss)/86400.0;
}


void jd2cal( int *yy, int *mo, int *dd, int *hh, int *mm, int *ss,
             double jd)
{
  int roundedSecs;
  long ijd;

  ijd = (long)(jd + 0.5);
  roundedSecs = (int) floor((jd + 0.5 - ijd)*86400 + 0.5);
  if (roundedSecs == 86400) {
    ijd++;
    roundedSecs = 0;
  }
  caldat( ijd, mo, dd, yy);
  *hh = roundedSecs/3600;
  roundedSecs -= 3600*(*hh);
  *mm = roundedSecs/60;
  *ss = roundedSecs - 60*(*mm);
}


void rdcal2jd( FILE *fp, double *jd)
{
  int c[6], i;

  for (i=0; i<=5; i++)
    c[i] = getint( fp);
  cal2jd( c[0], c[1], c[2], c[3], c[4], c[5], jd);
}


void wrtjd2cal( FILE *fp, double jd)
{
  int c[6];

  jd2cal( &c[0], &c[1], &c[2], &c[3], &c[4], &c[5], jd);
  fprintf( fp, "%4d %2d %2d %2d %2d %2d",
           c[0], c[1], c[2], c[3], c[4], c[5]);
}
