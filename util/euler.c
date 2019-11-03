#include "basic.h"

void euler2mat( double m[3][3], double phi, double theta, double psi)
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


void mat2euler( double m[3][3], double *phi, double *theta, double *psi)
{
  (*theta) = acos( m[2][2]);
  (*psi) = atan2( m[0][2], m[1][2]);
  (*phi) = atan2( m[2][0], -m[2][1]);
}

