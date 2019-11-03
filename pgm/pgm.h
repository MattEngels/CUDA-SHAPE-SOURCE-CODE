void pgmread( char *name, unsigned char **buf, int *nc, int *nr, int
			 *nlev);
void pgmwrite( char *name, unsigned char *buf, int nc, int nr, int nlev);
void ppmread( char *name, unsigned char **buf, int *nc, int *nr, int
			 *nlev);
void ppmwrite( char *name, unsigned char *buf, int nc, int nr, int nlev);
void wimaspgm( double **im, int xmin, int xmax, int ymin, int ymax, 
			  int clockwiserot, int xflip, int yflip,
			  char *name);
void wimaspgm0( double **im, int xmin, int xmax, int ymin, int ymax, 
			  int clockwiserot, int xflip, int yflip,
			  char *name);
void wimaspgmsc( double **im, int xmin, int xmax, int ymin, int ymax, 
			  double min, double max,
			  int clockwiserot, int xflip, int yflip,
			  char *name);
void wrasimaspgm0( double **im, int cmin, int cmax, int rmin, int rmax, 
			  char *name);
void wimasppm( double ***im, int xmin, int xmax, int ymin, int ymax, 
			  int clockwiserot, int xflip, int yflip,
			  char *name);
void wimasppm0( double ***im, int xmin, int xmax, int ymin, int ymax, 
			  int clockwiserot, int xflip, int yflip,
			  char *name);
void wimasppmsc( double ***im, int xmin, int xmax, int ymin, int ymax, 
			  double min, double max,
			  int clockwiserot, int xflip, int yflip,
			  char *name);
