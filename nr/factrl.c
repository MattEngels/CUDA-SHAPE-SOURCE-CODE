#include <math.h>

double factrl(int n)
{
	double gammln(double xx);
	void nrerror(char error_text[]);
	static int ntop=4;
	static double a[33]={1.0,1.0,2.0,6.0,24.0};
	int j;

	if (n < 0) nrerror("Negative factorial in routine factrl");
	if (n > 32) return exp(gammln(n+1.0));
	while (ntop<n) {
		j=ntop++;
		a[ntop]=a[j]*ntop;
	}
	return a[n];
}
