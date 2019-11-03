#define FOPEN(a,b,c) if ((a=fopen(b,c))==0) { printf("cannot open %s\n",b); exit(2); }

#define NEXTLINE(f) while ((getc(f)!=10)&&(!feof(f))) {}
