#include "basic.h"

void bcuint(double y[], double y1[], double y2[], double y12[], double x1l,
	double x1u, double x2l, double x2u, double x1, double x2, double *ansy,
	double *ansy1, double *ansy2)
{
	int i;
	double t,u,d1,d2,**c;

	c=matrix(1,4,1,4);
	d1=x1u-x1l;
	d2=x2u-x2l;
	bcucof(y,y1,y2,y12,d1,d2,c);
	if (x1u == x1l || x2u == x2l) nrerror("Bad input in routine bcuint");
	t=(x1-x1l)/d1;
	u=(x2-x2l)/d2;
	*ansy=(*ansy2)=(*ansy1)=0.0;
	for (i=4;i>=1;i--) {
		*ansy=t*(*ansy)+((c[i][4]*u+c[i][3])*u+c[i][2])*u+c[i][1];
		*ansy2=t*(*ansy2)+(3.0*c[i][4]*u+2.0*c[i][3])*u+c[i][2];
		*ansy1=u*(*ansy1)+(3.0*c[4][i]*t+2.0*c[3][i])*t+c[2][i];
	}
	*ansy1 /= d1;
	*ansy2 /= d2;
	free_matrix(c,1,4,1,4);
}
