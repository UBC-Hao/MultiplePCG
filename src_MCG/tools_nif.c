//gcc -o mat_nif.so -fpic -shared tools.c tools_nif.c poisson.c jacobi_cpu.c -llapack 
// mat:foo(mat:rndmat(3,3))
#include <erl_nif.h>
#include <stdio.h>
#include "tools.h"

void init_nif(){
    freopen("debug.txt","a",stdout);
    printf("===================");
}

ERL_NIF_TERM matrix2erl(ErlNifEnv* env, Matrix m){
    return enif_make_tuple3(env,enif_make_int(env,m.rows),enif_make_int(env,m.cols),enif_make_binary(env, &m.bin));
}

// Matlab style matrix representation dump.
void dumpMat(Matrix *m){
    printf("M = [");
    for(int i=0; i<m->rows; i++){
        for(int j = 0; j<m->cols; j++){
            printf("%.5lf", m->array[i+j*m->rows]);
            if(j != m->cols-1) printf(",");
        }
        if (i!=m->rows-1) printf(";");
    }
    printf("]\n");
    return;
}


int getMatrix(ErlNifEnv* env, ERL_NIF_TERM term, Matrix *m){
    int arity;
    const ERL_NIF_TERM* array;
    enif_get_tuple(env, term, &arity, &array);
    if ( arity!=3 ) return 0;
    enif_get_int(env, array[0], &(m->rows));
    enif_get_int(env, array[1], &(m->cols));
    enif_inspect_binary(env, array[2], &(m->bin));
    m->array = (double *)(m->bin.data);
    return 1;
}

Matrix createMat(int rows, int cols){
    ErlNifBinary bin;
    enif_alloc_binary(rows*cols*sizeof(double), &bin);
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.bin = bin;
    matrix.array = (double*) bin.data;
    return matrix;
}

Matrix zeros(int rows, int cols){
    Matrix matrix = createMat(rows,cols);
    memset(matrix.array,0,matrix.rows * matrix.cols * sizeof(double));
    return matrix;
}

Matrix matfromArray(int rows, int cols,double *data){
    Matrix matrix = createMat(rows,cols);
    memcpy(matrix.array,data,rows*cols*sizeof(double));
    return matrix;
}

void freeMat(Matrix *m){
    enif_release_binary(&m->bin);
}

Matrix rndmat(int rows, int cols){
    ErlNifBinary bin;
    enif_alloc_binary(rows*cols*sizeof(double), &bin);
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.bin = bin;
    matrix.array = (double*) bin.data;
    for (int i = 0;i<rows;i++){
        for (int j =0;j<cols;j++){
            matrix.array[i+j*rows] = randfrom(0.0,1.0);
        }
    }
    return matrix;
}

//foo(mat) in erl dumps the matrix mat
static ERL_NIF_TERM foo_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    //simple test function
    Matrix a;
    if (!getMatrix(env, argv[0], &a)) {
	    return enif_make_badarg(env);
    }
    dumpMat(&a);
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM add_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a,b;
    if (!getMatrix(env, argv[0], &a) || !getMatrix(env, argv[1], &b)) {
	    return enif_make_badarg(env);
    }
    double *Ap = (double *)malloc(a.rows*a.cols*sizeof(double));
    for (int i = 0;i<a.rows*a.cols;i++){
        Ap[i] = a.array[i]+b.array[i];
    }
    Matrix ret = matfromArray(a.rows,a.cols,Ap);
    free(Ap);
    return matrix2erl(env,ret);
}

static ERL_NIF_TERM sub_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a,b;
    if (!getMatrix(env, argv[0], &a) || !getMatrix(env, argv[1], &b)) {
	    return enif_make_badarg(env);
    }
    double *Ap = (double *)malloc(a.rows*a.cols*sizeof(double));
    for (int i = 0;i<a.rows*a.cols;i++){
        Ap[i] = a.array[i]-b.array[i];
    }
    Matrix ret = matfromArray(a.rows,a.cols,Ap);
    free(Ap);
    return matrix2erl(env,ret);
}

static ERL_NIF_TERM multiply_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a,b;
    if (!getMatrix(env, argv[0], &a) || !getMatrix(env, argv[1], &b)) {
	    return enif_make_badarg(env);
    }
    return matrix2erl(env,multiply(&a,&b));
}
static ERL_NIF_TERM tmultiply_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a,b;
    if (!getMatrix(env, argv[0], &a) || !getMatrix(env, argv[1], &b)) {
	    return enif_make_badarg(env);
    }
    return matrix2erl(env,tMultiply(&a,&b));
}

static ERL_NIF_TERM backslash_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a,b;
    if (!getMatrix(env, argv[0], &a) || !getMatrix(env, argv[1], &b)) {
	    return enif_make_badarg(env);
    }
    return matrix2erl(env,backslash(&a,&b));
}

static ERL_NIF_TERM merge_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a,b;
    if (!getMatrix(env, argv[0], &a) || !getMatrix(env, argv[1], &b)) {
	    return enif_make_badarg(env);
    }
    return matrix2erl(env,merge(&a,&b));
}


static ERL_NIF_TERM zeros_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    int m, n;
    if (!enif_get_int(env, argv[0], &m) || !enif_get_int(env, argv[1], &n)) {
	    return enif_make_badarg(env);
    }
    Matrix ret = zeros(m,n);
    return matrix2erl(env, ret);
}

static ERL_NIF_TERM vece_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    // returns an vector of size n by 1, and the m-th element is 1
    int m,n;
    if (!enif_get_int(env, argv[0], &m) || !enif_get_int(env, argv[1], &n)) {
	    return enif_make_badarg(env);
    }
    double *array = (double *)malloc(n*sizeof(double));
    memset(array,0,n*sizeof(double));
    array[m-1] = 1.0;
    Matrix ret = matfromArray(n,1,array);
    free(array);
    return matrix2erl(env, ret);
}

static ERL_NIF_TERM rndmat_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    int m, n;
    if (!enif_get_int(env, argv[0], &m) || !enif_get_int(env, argv[1], &n)) {
	    return enif_make_badarg(env);
    }
    Matrix ret = rndmat(m,n);
    return matrix2erl(env, ret);
}

static ERL_NIF_TERM premultiplyA_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a;
    if (!getMatrix(env, argv[0], &a)) {
	    return enif_make_badarg(env);
    }
    return matrix2erl(env,premultiplyA(&a));
}

static ERL_NIF_TERM premultiplyA_col_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a;
    int i;
    if ((!getMatrix(env, argv[0], &a)) || !enif_get_int(env, argv[1], &i)) {
	    return enif_make_badarg(env);
    }
    return matrix2erl(env,premultiplyA_col(&a,i));
}

static ERL_NIF_TERM inverse_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a;
    if (!getMatrix(env, argv[0], &a)) {
	    return enif_make_badarg(env);
    }
    return matrix2erl(env,inverse(&a));
}

static ERL_NIF_TERM norm_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Matrix a;
    if (!getMatrix(env, argv[0], &a)) {
	    return enif_make_badarg(env);
    }
    return enif_make_double(env,norm(&a));
}



static ERL_NIF_TERM initPDE_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    init_nif();
    int num;
    if (!enif_get_int(env, argv[0], &num)) {
	    return enif_make_badarg(env);
    }
    return initPoisson2D(env,num);
}

static ERL_NIF_TERM create_subdomain_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    int sx, sy, ex, ey;
    if (!enif_get_int(env, argv[0], &sx) || !enif_get_int(env, argv[1], &sy ) || !enif_get_int(env, argv[2], &ex)  || !enif_get_int(env, argv[3], &ey)) {
	    return enif_make_badarg(env);
    }
    return createSubdomain(env,sx,sy,ex,ey);
}

static ERL_NIF_TERM subsolve_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    //todo: change subdomain information read to one function
    int arity;
    const ERL_NIF_TERM* array;
    const ERL_NIF_TERM* loc_start;
    const ERL_NIF_TERM* loc_end;
    const ERL_NIF_TERM* matinfo;
    enif_get_tuple(env, argv[0], &arity, &array);

    enif_get_tuple(env, array[0], &arity, &loc_start);
    enif_get_tuple(env, array[1], &arity, &loc_end);
    enif_get_tuple(env, array[2], &arity, &matinfo);

    int startX, startY, endX, endY;
    enif_get_int(env, loc_start[0], &startX);
    enif_get_int(env, loc_start[1], &startY);

    enif_get_int(env, loc_end[0], &endX);
    enif_get_int(env, loc_end[1], &endY);

    Matrix A,B,b;
    getMatrix(env, matinfo[0] , &A);getMatrix(env, matinfo[1] , &B);getMatrix(env,argv[1],&b);
    return subdomain_solve(env, startX, startY, endX, endY, &A,&B,&b);
}

static ERL_NIF_TERM subrestrict_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    //todo: change subdomain information read to one function
    int arity;
    const ERL_NIF_TERM* array;
    const ERL_NIF_TERM* loc_start;
    const ERL_NIF_TERM* loc_end;
    const ERL_NIF_TERM* matinfo;
    enif_get_tuple(env, argv[0], &arity, &array);

    enif_get_tuple(env, array[0], &arity, &loc_start);
    enif_get_tuple(env, array[1], &arity, &loc_end);
    enif_get_tuple(env, array[2], &arity, &matinfo);

    int startX, startY, endX, endY;
    enif_get_int(env, loc_start[0], &startX);
    enif_get_int(env, loc_start[1], &startY);

    enif_get_int(env, loc_end[0], &endX);
    enif_get_int(env, loc_end[1], &endY);

    Matrix b;
    getMatrix(env,argv[1],&b);
    return subdomain_restrict(env, startX, startY, endX, endY, &b);
}

static ERL_NIF_TERM subprolong_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    //todo: change subdomain information read to one function
    int arity;
    const ERL_NIF_TERM* array;
    const ERL_NIF_TERM* loc_start;
    const ERL_NIF_TERM* loc_end;
    const ERL_NIF_TERM* matinfo;
    enif_get_tuple(env, argv[0], &arity, &array);

    enif_get_tuple(env, array[0], &arity, &loc_start);
    enif_get_tuple(env, array[1], &arity, &loc_end);
    enif_get_tuple(env, array[2], &arity, &matinfo);

    int startX, startY, endX, endY;
    enif_get_int(env, loc_start[0], &startX);
    enif_get_int(env, loc_start[1], &startY);

    enif_get_int(env, loc_end[0], &endX);
    enif_get_int(env, loc_end[1], &endY);

    Matrix b;
    getMatrix(env,argv[1],&b);
    return subdomain_prolongate(env, startX, startY, endX, endY, &b);
}


static ErlNifFunc nif_funcs[] = {
    {"foo", 1, foo_nif},
    {"zeros", 2, zeros_nif},
    {"rndmat",2,rndmat_nif},
    {"add",2,add_nif},
    {"sub",2,sub_nif},
    {"multiply",2,multiply_nif},
    {"tmultiply",2,tmultiply_nif},
    {"backslash",2,backslash_nif},
    {"merge",2,merge_nif},
    {"initpde",1,initPDE_nif},
    {"amul",1,premultiplyA_nif},
    {"amul",2,premultiplyA_col_nif},
    {"vece",2,vece_nif},
    {"norm",1,norm_nif},
    {"subcreate",4,create_subdomain_nif},
    {"subsolve",2,subsolve_nif},
    {"subrestrict",2,subrestrict_nif},
    {"subprolong",2,subprolong_nif},
    {"inverse",1,inverse_nif}
};


ERL_NIF_INIT(mat, nif_funcs, NULL, NULL, NULL, NULL);