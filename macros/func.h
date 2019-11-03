/* Modified 2007 August 19 by CM: put parentheses around MIN and MAX arguments */
/* Modified 2009 November 15 by CM: fixed sequence-point ambiguity in SINC2 macro */

#define MIN( a, b) ( ( (a) < (b) ) ? (a) : (b) )
#define MAX( a, b) ( ( (a) > (b) ) ? (a) : (b) )
#define SINC2(x) ( ( tmp=fabs(3.141592653589793*(x)) ) > 1.0e-4 ? ( tmp=sin(tmp)/tmp, tmp*tmp ) : 1.0 )
#define TRI(x) ( ( tmp=(1-fabs(x)) ) > 0.0 ? tmp : 0.0 )
#define TRI2(x) ( ( tmp=(1-fabs(x)) ) > 0.0 ? tmp*tmp : 0.0 )
