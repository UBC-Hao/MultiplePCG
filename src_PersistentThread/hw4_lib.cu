#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "hw4_lib.h"

// _cudaTry -- function used by the cudaTry macro (see hw3_lib.h)
void _cudaTry(cudaError_t cudaStatus, const char *fileName, int lineNumber) {
  if(cudaStatus != cudaSuccess) {
    fprintf(stderr, "%s in %s line %d\n",
        cudaGetErrorString(cudaStatus), fileName, lineNumber);
    exit(1);
  }
}

// usage(i, opt_names, pos) -- report a command-line usage error
//   i is the index of argv that caused the error.
//   opt_names is an NULL-terminated array of strings for the option names.
//     opt_names[0] is the name of the program.
//     opt_names[i] is a descriptive name for argv[i]
//   pos: if true, the argument must be a postive integer.
//        otherwise, the argument must be a non-negative integer.
void usage(int i, const char **opt_names, uint pos) {
  uint n_opt;
  for(n_opt = 0; (n_opt < 20) && opt_names[n_opt]; n_opt++);
  fprintf(stderr, "usage:");
  for(uint j = 0; j <= i; j++)
    if(j < n_opt) fprintf(stderr, " %s", opt_names[j]);
    else fprintf(stderr, " !unknown-option!");
  if(i+1 < n_opt) {
    fprintf(stderr, " [");
    for(uint j = i+1; j < n_opt; j++)
      fprintf(stderr, "%s%s", j > i+1 ? " " : "", opt_names[j]);
    fprintf(stderr, "]");
  }
  fprintf(stderr, "\n");
  fprintf(stderr, "  %s must be a %s integer\n",
          opt_names[i], pos ? "positive" : "non-negative");
  exit(-1);
}

// get_uint(argc, argv, opt_names, i, v0, pos): return the i^th command line argument as an unsigned int.
//   opt_names is an array of strings giving the name of the program
//     and descriptive names of the arguments (for error messages).
//   v0 is the default value to return if argc < i.
//   pos: if true, the value most be strictly positive.
//     otherwise, we expect a non-negative integer.
uint get_uint(int argc, char **argv, const char **opt_names, int i, uint v0, uint pos) {
  if(argc <= i) return(v0);
  if(argv[i][0] == '-') return(v0);
  for(int j = 0; argv[i][j]; j++) {
    if(!isdigit(argv[i][j]))
      usage(i, opt_names, pos);
  }
  int v = atoi(argv[i]);
  if((v < 0) || (pos && v == 0)) usage(i, opt_names, pos);
  return((uint)(v));
}


// mean(n, sum): return the mean of n samples where sum is the sum of the samples.
double mean(double sum, uint n) {
  return(sum/n);
}

// stdev(sum, sum_sq, n): return the standard-deviation of n samples
//   sum: the sum of the samples
//   sum_sq: the sum of the squares of the samples
//   n: the number of samples
// Remark: there are other formulation that are less susceptible to round-off
//   if stdev << mean, but we should't be pushing extreme cases in CPSC 418.
double stdev(double sum, double sum_sq, uint n) {
  return(sqrt((sum_sq - sum*sum/n)/(n-1)));
}

// rms(data, n):
//   The square root of the mean of the squares of the elements of data.
//   n is the number of elewments of data.
double rms(float *data, uint n) {
  double sum = 0.0;
  for(int i = 0; i < n; i++)
    sum += (data[i] * data[i]);
  return(sqrt(sum/n));
}

// argmax(data, n):
//   Find the element of data with the largest absolute value.  Return the
//   index of the first such element.  n is the number of elements of data.
//   If n==0, then data is empty, and we return -1.
int argmax(float *data, uint n) {
  double max = 0.0;
  int imax = -1;
  for(int i = 0; i < n; i++) {
    if(abs(data[i]) > max) {
      max = data[i];
      imax = i;
    }
  }
  return(imax);
}


// rand_vector(v, n)
//   generate n random values and store them in the vector n.
//   The distribution here is choses so that most point are in the
//   convergence region of the Hénon map -- i.e. repeatedly applying the
//   map doesn't head off to infinity.  I should probably write a more
//   general purpose version some day.
void rand_vector(float *v, uint n) {
  for(uint i = 0; i < n; i++)
    v[i] = 0.6 * ((float)rand() / (float)RAND_MAX - 0.5);
}
