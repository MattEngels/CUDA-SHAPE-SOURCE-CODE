/*****************************************************************************************
                                                                                  init.c

Displays the version of shape that will run, the starting time, and the root node's PID
and hostname, checks that at least one command-line argument was provided, and initializes
some variables related to parallel processing

Modified July 5, 2016 for use in shape-cuda
*****************************************************************************************/

#include "head.h"

void init( int argc, char *argv[], char *progname)
{
  char message[MAXLEN], commandline[MAXLEN], localtimestring[MAXLEN], hostname[MAXLEN];
  int i;
  long pid_long;
  pid_t pid;
  

  /*  Display command line, program version, PID, and hostname,
      and check there's at least one command-line argument            */

  printf("# %s version %s build %s", progname, VERSION, BUILD);

  if (argc == 1) {
	sprintf(message, "usage - %s par mod [obs]\n", progname);
	bailout(message);
  }
  printf("#\n");
  strcpy(commandline, "# command line:");
  for (i = 0; i < argc; i++) {
	strcat(commandline, " ");
	strcat(commandline, argv[i]);
  }
  printf("%s\n", commandline);
  printf("#\n");
  timestring(localtimestring, MAXLEN, "");
  printf("# starting time %s\n", localtimestring);
  pid = getpid();
  pid_long = (long) pid; /* Assumes pid_t fits in a long */
  //(void) gethostname(hostname, MAXLEN - 1);
  printf("#\n");
  printf("# node  0 running as pid %ld\n", pid_long);

}
