/*  direxists and createdir written 2004 December 1 by CM  */

#include "basic.h"
#include <sys/stat.h>
#include <errno.h>


/*  Boolean function which says whether or not a directory exists;  */
/*  treat the null string as equivalent to "." (current directory)  */

int direxists(char *dirname)
{
  struct stat sts;

  if (!strcmp(dirname, ""))
    return 1;
  else if (stat(dirname, &sts) == -1 && errno == ENOENT)
    return 0;
  else
    return 1;
}


/*  Create a directory: return 1 if successful, 0 if unsuccessful,  */
/*                      or -1 if the directory already exists       */

int createdir(char *dirname)
{
  char cmd[256];

  if (direxists(dirname)) {
      return -1;
  } else {
      sprintf( cmd, "\\mkdir -p %s", dirname);
      system( cmd);
      return direxists(dirname);
  }
}
