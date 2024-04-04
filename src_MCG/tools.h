#include <erl_nif.h>
#include <string.h>
#include <cblas.h>


typedef struct{
    int rows;
    int cols;
    double *array;//matrix is stored as col major order same as blas
    ErlNifBinary bin;
} Matrix;

double randfrom(double min, double max);
Matrix backslash(Matrix *A, Matrix *B);
Matrix transpose(Matrix *A);
Matrix multiply(Matrix *A, Matrix *B);
Matrix tMultiply(Matrix *A, Matrix *B);
Matrix multiplyT(Matrix *A, Matrix *B);
Matrix merge(Matrix *A, Matrix *B);
Matrix matfromArray(int rows, int cols,double *data);
ERL_NIF_TERM initPoisson2D(ErlNifEnv* env,int num);
Matrix premultiplyA(Matrix *m);
double norm(Matrix *mat);
ERL_NIF_TERM matrix2erl(ErlNifEnv* env, Matrix m);
ERL_NIF_TERM createSubdomain(ErlNifEnv* env, int startX,int startY,int endX,int endY);
ERL_NIF_TERM subdomain_solve(ErlNifEnv* env, int startX, int startY, int endX, int endY, Matrix *A_sub, Matrix *B_sub, Matrix *b);
ERL_NIF_TERM subdomain_restrict(ErlNifEnv* env, int startX, int startY, int endX, int endY, Matrix *b);
ERL_NIF_TERM subdomain_prolongate(ErlNifEnv* env, int startX, int startY, int endX, int endY, Matrix *b);
Matrix premultiplyA_col(Matrix *m, int i);
Matrix inverse(Matrix *A);