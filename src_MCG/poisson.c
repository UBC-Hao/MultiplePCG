#include "tools.h"

//A(i,j) represents the point (x_i,y_j). 
#define A(i,j) A[(i)+(j)*m]
#define B(i,j) B[(i)+(j)*m]
#define v(i,j) v[(i)+(j)*m]
#define v_sub(i,j) v[(i)+(j)*m_sub]
#define b(i,j) b[(i)+(j)*m]


//initpde returns some infomation for erlang

// A B is an coefficient matrix,
double *A;
double *B;

// M N is the problem size. i.e. the 2D grid's Length and Width
int M,N;
double H;

//initial right hand side vector b, we'll always set the rhs to be ones*h^2 for simplicity
Matrix initial_rhs(){
    int n = N-1;
    int m = M-1;
    double *b = (double *)malloc(m*n*sizeof(double));
    for (int j = 0;j<n;j++){
        for (int i = 0;i<m;i++){
            b(i,j) =  H*H;
        }
    }
    Matrix mat = matfromArray(m*n,1,b);
    free(b);
    return mat;
}

ERL_NIF_TERM initPoisson2D(ErlNifEnv* env,int num){

    //set the problem size and the coeeficient matrix
    //M = N, for simplicity, other wise it's just a matter of scaling.
    M = num;// the grid Width
    N = M;// the grid Height
    H = 1.0/N;
    int MaxItrs = 100;
    double Tol = 1e-5;
    int n = N-1;
    int m = M-1;
    A = (double *)malloc(m*n*sizeof(double));
    B = (double *)malloc(m*n*sizeof(double));
    for(int j = 0;j<N-1;j++){
        for(int i = 0;i<M-1;i++){
            A(i,j) = 1; 
            B(i,j) = 1;
        }
    }
    return enif_make_tuple3(env,
    enif_make_tuple2(env,enif_make_int(env,M), enif_make_int(env,N)),
    enif_make_tuple2(env,enif_make_int(env,MaxItrs), enif_make_double(env, Tol)),
    matrix2erl(env,initial_rhs()));
}


extern void amul(double *xx, double *yy, int M, int N,double a_c,double b_c);
// matrix-free implementation of multiply
// input a vector v
// output A*v
double * premultiplyA_array(double *v){
    int n = N - 1;// the actual interior grids height
    int m = M - 1;// the actual interior grids width
    double *b = (double *)malloc(m*n*sizeof(double));// free
    amul(v,b,M,N,A(1,1),B(1,1));
    /*for (int j = 0;j<n;j++){
        for (int i = 0;i<m;i++){
            double aa = A(i,j);
            double bb = B(i,j);
            double center = 2*(aa+bb)*v(i,j);
            double up=0,down=0,left=0,right=0;
            if (j<n-1) up = -bb*v(i,j+1);
            if (j>0) down = -bb*v(i,j-1);
            if (i>0) left = -aa*v(i-1,j);
            if (i<m-1) right = -aa*v(i+1,j);
            b(i,j) = center+up+down+left+right;
        }
    }*/
    return b;
}



// matrix-free implementation of multiply, optimization for zeros patterns of vector
// input a vector v_tilde a vector in subdomain
// output A*v
double * premultiplyA_array_sub(double *v, int startX, int startY, int endX, int endY){
    int n = N - 1;// the actual interior grids height
    int m = M - 1;// the actual interior grids width
    int m_sub = endX - startX + 1;
    int n_sub = endY - startY + 1;
    double *b = (double *)malloc(m*n*sizeof(double));// free
    memset(b,0,m*n*sizeof(double));
    for (int j = 0;j<n_sub;j++){
        for (int i = 0;i<m_sub;i++){
            double aa = A(i+startX,j+startY);
            double bb = B(i+startX,j+startY);
            double pval = 2*(aa+bb)*v_sub(i,j);
            if (j<n_sub-1) pval += -bb*v_sub(i,j+1);
            if (j>0) pval += -bb*v_sub(i,j-1);
            if (i>0) pval += -aa*v_sub(i-1,j);
            if (i<m_sub-1) pval += -aa*v_sub(i+1,j);
            b(startX+i,startY+j) = pval;
        }
    }
    return b;
}

//input a matrix M, output A*M
Matrix premultiplyA(Matrix *m){
    int rows = m->rows;
    int cols = m->cols;
    double *data = m->array;
    double *results = (double *)malloc(rows*cols*sizeof(double));
    //if (rows == (M-1)*(N-1))
    for (int i = 0;i<cols;i++){
        double *col_i = premultiplyA_array(data+i*rows);
        memcpy(results+rows*i, col_i, rows*sizeof(double));
        free(col_i);
    }
    Matrix mat = matfromArray(rows,cols,results);
    free(results);
    return mat;
}


//input a matrix M, output A*M(:,i), i = 0,1,2,3..
Matrix premultiplyA_col(Matrix *m, int i){
    int rows = m->rows;
    int cols = m->cols;
    double *data = m->array;
    double *results = (double *)malloc(rows*1*sizeof(double));

    double *col_i = premultiplyA_array(data+i*rows);
    memcpy(results, col_i, rows*sizeof(double));
    free(col_i);
    
    Matrix mat = matfromArray(rows,1,results);
    free(results);
    return mat;
}

//returns the subdomain information,
// (startX,startY) -> (endX,endY) select a certain subdomain,
// the indices are the indices of the interior grids
ERL_NIF_TERM createSubdomain(ErlNifEnv* env, int startX,int startY,int endX,int endY)
{
    int m_sub = endX - startX + 1;
    int n_sub = endY - startY + 1;
    int M_sub = m_sub + 1;
    int N_sub = n_sub + 1;
    double *A_sub = (double *)malloc(m_sub*n_sub*sizeof(double));
    double *B_sub = (double *)malloc(m_sub*n_sub*sizeof(double));
    //copy part of A,B to A_sub,B_sub
    for(int i = 0;i<n_sub;i++){
        memcpy(A_sub+m_sub*i,A+(M-1)*(startY+i)+startX,m_sub*sizeof(double));
        memcpy(B_sub+m_sub*i,B+(M-1)*(startY+i)+startX,m_sub*sizeof(double));
    }
    Matrix matA = matfromArray(m_sub*n_sub,1,A_sub);
    Matrix matB = matfromArray(m_sub*n_sub,1,B_sub);
    free(A_sub);free(B_sub);
    return  enif_make_tuple3(env, 
            enif_make_tuple2(env,enif_make_int(env,startX),enif_make_int(env,startY)),
            enif_make_tuple2(env,enif_make_int(env,endX),enif_make_int(env,endY)),
            enif_make_tuple2(env,matrix2erl(env,matA),matrix2erl(env,matB)));
}

extern void subdomain_solve_cpu(int m, int n, double *A, double *B, double *b, double *ans);
extern void subdomain_solve_gpu(int m, int n, double *A, double *B, double *b, double *ans);

ERL_NIF_TERM subdomain_solve(ErlNifEnv* env, int startX, int startY, int endX, int endY, Matrix *A_sub, Matrix *B_sub, Matrix *b){
    int m_sub = endX - startX + 1;
    int n_sub = endY - startY + 1;
    double *sub_ans =  (double *)malloc(m_sub*n_sub*sizeof(double));
    subdomain_solve_gpu(m_sub,n_sub,A_sub->array,B_sub->array,b->array,sub_ans);//GPU code must be run on multiple machines.
    //subdomain_solve_cpu(m_sub,n_sub,A_sub->array,B_sub->array,b->array,sub_ans);//CPU code can be run on a single machine if neeeded.
    Matrix mat = matfromArray(m_sub*n_sub,1,sub_ans);
    free(sub_ans);
    return matrix2erl(env,mat);
}

// restrict matrix b in the whole domain into subdomain
ERL_NIF_TERM subdomain_restrict(ErlNifEnv* env, int startX, int startY, int endX, int endY, Matrix *b){
    int m_sub = endX - startX + 1;
    int n_sub = endY - startY + 1;
    double *ret = (double *)malloc(m_sub*n_sub*sizeof(double));
    for(int i = 0;i<n_sub;i++) memcpy(ret+m_sub*i,(b->array)+(M-1)*(startY+i)+startX,m_sub*sizeof(double));
    Matrix mat = matfromArray(m_sub*n_sub,1,ret);
    free(ret);
    return matrix2erl(env,mat);
}

// prolongate matrix b in the subdomain into whole domain
ERL_NIF_TERM subdomain_prolongate(ErlNifEnv* env, int startX, int startY, int endX, int endY, Matrix *b){
    int m_sub = endX - startX + 1;
    int n_sub = endY - startY + 1;
    double *ret = (double *)malloc((M-1)*(N-1)*sizeof(double));
    memset(ret,0,(M-1)*(N-1)*sizeof(double));
    for(int i = 0;i<n_sub;i++) memcpy(ret+(M-1)*(startY+i)+startX,(b->array)+m_sub*i,m_sub*sizeof(double));
    Matrix mat = matfromArray((M-1)*(N-1),1,ret);
    free(ret);
    return matrix2erl(env,mat);
}





