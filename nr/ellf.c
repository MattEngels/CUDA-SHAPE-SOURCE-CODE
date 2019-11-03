#include "basic.h"

double ellf(double phi, double ak)
{
	double rf(double x, double y, double z);
	double s;

	s=sin(phi);
	return s*rf(DSQR(cos(phi)),(1.0-s*ak)*(1.0+s*ak),1.0);
}
