/***************************************************************************

								gray2rgb.c

converts a gray-level (0 to 1) into false color rgb values (0 to 1).

***************************************************************************/

#include "basic.h"

void gray2rgb( double gray, double *r, double *g, double *b)
{
  double svnth;

  svnth = 1.0/7.0;

  /* blue value */
  if (gray<0.0)
	*b = 0.0;
  if ((gray>=0*svnth)&&(gray<1*svnth))
	*b = 7*gray;
  if ((gray>=1*svnth)&&(gray<2*svnth))
	*b = 1.0;
  if ((gray>=2*svnth)&&(gray<3*svnth))
	*b = 3-7*gray;
  if ((gray>=3*svnth)&&(gray<4*svnth))
	*b = 0.0;
  if ((gray>=4*svnth)&&(gray<5*svnth))
	*b = 0.0;
  if ((gray>=5*svnth)&&(gray<6*svnth))
	*b = 7*gray-5;
  if ((gray>=6*svnth)&&(gray<7*svnth))
	*b = 1.0;
  if (gray>1.0)
	*b = 1.0;

  /* green value */
  if (gray<0.0)
	*g = 0.0;
  if ((gray>=0*svnth)&&(gray<1*svnth))
	*g = 0.0;
  if ((gray>=1*svnth)&&(gray<2*svnth))
	*g = 7*gray-1;
  if ((gray>=2*svnth)&&(gray<3*svnth))
	*g = 1.0;
  if ((gray>=3*svnth)&&(gray<4*svnth))
	*g = 1.0;
  if ((gray>=4*svnth)&&(gray<5*svnth))
	*g = 5-7*gray;
  if ((gray>=5*svnth)&&(gray<6*svnth))
	*g = 0.0;
  if ((gray>=6*svnth)&&(gray<7*svnth))
	*g = 7*gray-6;
  if (gray>1.0)
	*g = 1.0;

  /* green value */
  if (gray<0.0)
	*r = 0.0;
  if ((gray>=0*svnth)&&(gray<3*svnth))
	*r = 0.0;
  if ((gray>=3*svnth)&&(gray<4*svnth))
	*r = 7*gray-3;
  if ((gray>=4*svnth)&&(gray<7*svnth))
	*r = 1.0;

  if (gray>1.0)
	*r = 1.0;

}
