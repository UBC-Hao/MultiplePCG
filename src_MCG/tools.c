/* complex.c */
#include "tools.h"

// Lapack functions
void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
void dgetri_(int *N, double *A, int *LDA, int *IPIV, double *WORK, int* LWORK, int* INFO);
void dgesv_(int* N,int *NRHS,double *A,int *LDA,int *IPIV,double *B,int *LDB,int *INFO);
void dgesvx_(	char *FACT,char *TRANS,int *N,int *NRHS,double *A,int *LDA,double *AF,int *LDAF,int *IPIV,char *EQUED,double *R,double *C,double *B,int *LDB,double *X,int *LDX,double *RCOND,double *FERR,double *BERR,double *WORK,int *IWORK,int *INFO 
);
double dnrm2_(int *N, double *V, int *INC);
// end

double norm(Matrix *mat){
  // mat is of size N , 1
  int N = mat->rows;
  int Inc = 1;
  return dnrm2_(&N,mat->array,&Inc);
}

double randfrom(double min, double max)
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void copyMatrixElements(Matrix *A,double *Ap){
  // in blas, matrix is storesd in col major order
  memcpy(Ap,A->array,(A->rows)*(A->cols)*sizeof(double));
}

extern void cublas_Dgemm(int transa, int transb,
                           int m, int n, int k,
                           double          alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           double          beta,
                           double          *C, int ldc,
                           int A_length,int B_length,int result_length);

//blas wrapper for multiplication, returns A*B
Matrix multiply_(Matrix *A, Matrix *B,CBLAS_TRANSPOSE transA,CBLAS_TRANSPOSE transB){
  int rows = A->rows;
  int cols = B->cols;
  double *result;
  // Todo: if possible, other matrices multiplications
  if (transA==CblasTrans&&transB==CblasNoTrans){
    rows = A->cols;
    cols = B->cols;
    result = (double*)malloc(rows*cols*sizeof(double));
    //cblas_dgemm(CblasColMajor,transA,transB,A->cols,B->cols,A->rows,1.0,A->array,A->rows,B->array,B->rows,0.0,result,rows);
    cublas_Dgemm(1,0,A->cols,B->cols,A->rows,1.0,A->array,A->rows,B->array,B->rows,0.0,result,rows,A->rows*A->cols,B->rows*B->cols,rows*cols);
  }else{
    result = (double*)malloc(rows*cols*sizeof(double));
    //cblas_dgemm(CblasColMajor,transA,transB,A->rows,B->cols,A->cols,1.0,A->array,A->rows,B->array,B->rows,0.0,result,rows);
    cublas_Dgemm(0,0,A->rows,B->cols,A->cols,1.0,A->array,A->rows,B->array,B->rows,0.0,result,rows,A->rows*A->cols,B->rows*B->cols,rows*cols);
  }
    Matrix ret = matfromArray(rows,cols,result);
  free(result);
  return ret;
}
//returns A*B
Matrix multiply(Matrix *A, Matrix *B){
  return multiply_(A,B,CblasNoTrans,CblasNoTrans);
}
//returns A'*B
Matrix tMultiply(Matrix *A, Matrix *B){
  return multiply_(A,B,CblasTrans,CblasNoTrans);
}
//returns A*B'
Matrix multiplyT(Matrix *A, Matrix *B){
  return multiply_(A,B,CblasNoTrans,CblasTrans);
}
// returns A\B, the code is a wrapper for lapack function, dgesv_ 
Matrix backslash(Matrix *A, Matrix *B){
  int *N = &A->rows;
  int *NRHS = &B->cols;
  double *Ap = (double *)malloc((A->rows)*(A->cols)*sizeof(double));
  copyMatrixElements(A,Ap);
  int *IPIV = (int *)malloc((A->rows)*sizeof(int));
  double *Bp = (double *)malloc((B->rows)*(B->cols)*sizeof(double));
  copyMatrixElements(B,Bp);
  int INFO;
  dgesv_(N,NRHS,Ap,N,IPIV,Bp,N,&INFO);
  free(Ap); //BP should be the solution
  Matrix ret = matfromArray(B->rows,B->cols,Bp);
  free(Bp);
  free(IPIV);
  return ret;
}

// returns A^(-1), the code is a wrapper for lapack function 
Matrix inverse(Matrix *A){
  int N = A->rows;
  double *Ap = (double *)malloc((A->rows)*(A->cols)*sizeof(double));
  double *Work = (double *)malloc((A->rows)*(A->cols)*sizeof(double));
  copyMatrixElements(A,Ap);
  int *IPIV = (int *)malloc((A->rows)*sizeof(int));
  int INFO;
  int LWORK = N*N;
  dgetrf_(&N,&N,Ap,&N,IPIV,&INFO);
  dgetri_(&N,Ap,&N,IPIV,Work,&LWORK,&INFO);
  Matrix ret = matfromArray(A->rows,A->cols,Ap);
  free(Ap); //BP should be the solution
  free(IPIV); free(Work); 
  return ret;
}


//returns A^T
Matrix transpose(Matrix *A){
  double *Ap = (double *)malloc((A->rows)*(A->cols)*sizeof(double));
  for (int j =0;j<(A->cols);j++){
    for (int i =0;i<(A->rows);i++){
      Ap[j+i*(A->cols)] = A->array[i+j*(A->rows)];
    }
  }
  Matrix ret = matfromArray(A->cols,A->rows,Ap);
  free(Ap);
  return ret;
}

//returns [A|B], merge A and B by column
Matrix merge(Matrix *A,Matrix *B){
  double *Ap = (double *)malloc((A->rows)*(A->cols+B->cols)*sizeof(double));
  memcpy(Ap,A->array,(A->rows)*(A->cols)*sizeof(double));
  memcpy(Ap+(A->rows)*(A->cols),B->array,(B->rows)*(B->cols)*sizeof(double));
  Matrix ret = matfromArray(A->rows,A->cols+B->cols,Ap);
  free(Ap);
  return ret;
}
