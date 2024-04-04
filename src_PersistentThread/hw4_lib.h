
void _cudaTry(cudaError_t cudaStatus, const char *fileName, int lineNumber);
void usage(int i, const char **opt_names, uint pos) ;
uint get_uint(int argc, char **argv, const char **opt_names, int i, uint v0, uint pos) ;
double mean(double sum, uint n);
double stdev(double sum, double sum_sq, uint n);
double rms(float *data, uint n);
int argmax(float *data, uint n) ;
void rand_vector(float *v, uint n);