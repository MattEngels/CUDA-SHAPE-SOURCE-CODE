extern "C" {
#include "../shape/head.h"
}

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);

__device__ void dev_jacobi(double a[3][3], int n, double d[3], double v[3][3], int *nrot)
{
	int j,iq,ip,i;
	double tresh,theta,tau,t,sm,s,h,g,c;

	double b[3];
	double z[3];

	for (ip=0;ip<3;ip++) {
		for (iq=0;iq<3;iq++) v[ip][iq]=0.0;
		v[ip][ip]=1.0;
	}
	for (ip=0;ip<3;ip++) {
		b[ip]=d[ip]=a[ip][ip];
		z[ip]=0.0;
	}
	*nrot=0;
	for (i=0;i<50;i++) {
		sm=0.0;
		for (ip=0;ip<3-1;ip++) {
			for (iq=ip+1;iq<3;iq++)
				sm += fabs(a[ip][iq]);
		}
		if (sm == 0.0)
			return;
		if (i < 4)
			tresh=0.2*sm/(9);
		else
			tresh=0.0;
		for (ip=0;ip<2;ip++) {
			for (iq=ip+1;iq<3;iq++) {
				g=100.0*fabs(a[ip][iq]);
				if (i > 4 && fabs(d[ip])+g == fabs(d[ip])
					&& fabs(d[iq])+g == fabs(d[iq]))
					a[ip][iq]=0.0;
				else if (fabs(a[ip][iq]) > tresh) {
					h=d[iq]-d[ip];
					if (fabs(h)+g == fabs(h))
						t=(a[ip][iq])/h;
					else {
						theta=0.5*h/(a[ip][iq]);
						t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
						if (theta < 0.0) t = -t;
					}
					c=1.0/sqrt(1+t*t);
					s=t*c;
					tau=s/(1.0+c);
					h=t*a[ip][iq];
					z[ip] -= h;
					z[iq] += h;
					d[ip] -= h;
					d[iq] += h;
					a[ip][iq]=0.0;
					for (j=0;j<ip-1;j++) {
						ROTATE(a,j,ip,j,iq)
					}
					for (j=ip+1;j<=iq-1;j++) {
						ROTATE(a,ip,j,j,iq)
					}
					for (j=iq+1;j<=n;j++) {
						ROTATE(a,ip,j,iq,j)
					}
					for (j=0;j<n;j++) {
						ROTATE(v,j,ip,j,iq)
					}
					++(*nrot);
				}
			}
		}
		for (ip=0;ip<3;ip++) {
			b[ip] += z[ip];
			d[ip]=b[ip];
			z[ip]=0.0;
		}
	}
	printf("Too many iterations in routine JACOBI");
}
#undef ROTATE
