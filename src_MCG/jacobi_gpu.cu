#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include "cublas_v2.h"

#define BLOCK_SIZE 32
#define OUT_TILE_SIZE ((BLOCK_SIZE) - 2)
#define A_s(i,j) A_s[j][i]
#define B_s(i,j) B_s[j][i]


// Kernel - jacobi iteration
// This method uses dynamical coefficient matrices A and B, which will introduce a lot of global memory access
// and cannot be stored in constant memory. CGMA is small.
__global__ void jacobi_itrs(float *A, float *B, float *rhs,float *xx1, float *xx2, int M, int N)
{
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    //halo cells
    int idx = bx * OUT_TILE_SIZE + tx - 1; int idy = by * OUT_TILE_SIZE + ty - 1;


    //Too big to put in the constant memory
    __shared__ float A_s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float rhs_s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float xx1_s[BLOCK_SIZE][BLOCK_SIZE];

    //Load A and B into shared memory
    // The indexing is for linearly indexing the 2D array
    if ( 0<=idy&& idy < N+1 && 0<=idx&&idx < M+1){ //boundary check
        A_s[ty][tx] = A[idx + idy * (M+1)];
        B_s[ty][tx] = B[idx + idy * (M+1)];
        rhs_s[ty][tx] = rhs[idx + idy * (M+1)];
        xx1_s[ty][tx] = xx1[idx + idy * (M+1)];
    }else{
        A_s[ty][tx] = 0;
        B_s[ty][tx] = 0;
        rhs_s[ty][tx] = 0;
        xx1_s[ty][tx] = 0;
    }
    
    __syncthreads();

    //boundary check
    if (0<idx && idx < M && 0<idy && idy < N & tx<BLOCK_SIZE-1 & ty<BLOCK_SIZE-1 & tx>0 & ty>0){
        float aa = A_s(tx,ty);
        float bb = B_s(tx,ty);
        xx2[idx + idy * (M+1)] = (rhs_s[ty][tx] + bb*(xx1_s[ty+1][tx] + xx1_s[ty-1][tx]) + aa*(xx1_s[ty][tx+1] + xx1_s[ty][tx-1]))/(2*(aa+bb));
    }
}

// Kernel - jacobi iteration
// This method uses fixed coefficient a_c and b_c 
__global__ void jacobi_itrs_fixed(float *rhs, float *xx1, float *xx2, int M, int N, float aa, float bb)
{
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    //halo cells
    int idx = bx * OUT_TILE_SIZE + tx - 1; int idy = by * OUT_TILE_SIZE + ty - 1;

    //Too big to put in the constant memory
    __shared__ float rhs_s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float xx1_s[BLOCK_SIZE][BLOCK_SIZE];

    //Load A and B into shared memory
    // The indexing is for linearly indexing the 2D array
    if ( 0<=idy&& idy < N+1 && 0<=idx&&idx < M+1){ //boundary check
        rhs_s[ty][tx] = rhs[idx + idy * (M+1)];
        xx1_s[ty][tx] = xx1[idx + idy * (M+1)];
    }else{
        rhs_s[ty][tx] = 0;
        xx1_s[ty][tx] = 0;
    }
    
    __syncthreads();
    //boundary check
    if (0<idx && idx < M && 0<idy && idy < N & tx<BLOCK_SIZE-1 & ty<BLOCK_SIZE-1 & tx>0 & ty>0){
        xx2[idx + idy * (M+1)] = (rhs_s[ty][tx] + bb*(xx1_s[ty+1][tx] + xx1_s[ty-1][tx]) + aa*(xx1_s[ty][tx+1] + xx1_s[ty][tx-1]))/(2*(aa+bb));
    }
}


extern "C" void subdomain_solve_gpu(int m, int n, double *A, double *B, double *b,double *worker){
    //constant or variable coefficient, 0 for constant, 1 for variable
    int tag = 0;

    //iteration vectors used in the Jacobi method
    float *xx1_d,*xx2_d;
    float a_c,b_c;
    float *xx_tmp; // tmp varible to store solution

    // Domain infomation
    int M_g,N_g;
    float *b_g;
    float *A_d,*B_d,*rhs_d,*Initial;

    M_g = m + 1; N_g = n + 1;// m is interior nodes number, M will be total interval number
    if (tag == 0){
        //with fixed coefficient
        a_c = A[1]; b_c = B[1];
    }else{
        //with variable coefficient
        //setSubdomain_info
        //A_g is an array that contains the boundary as well,
        float *A_g;
        float *B_g;
        //we do this to reduce control divergence.
        A_g = (float*)malloc((M_g+1) * (N_g+1) * sizeof(float));
        B_g = (float*)malloc((M_g+1) * (N_g+1) * sizeof(float));
        memset(A_g, 0, (M_g+1) * (N_g+1) * sizeof(float));
        memset(B_g, 0, (M_g+1) * (N_g+1) * sizeof(float));
        // only update interior points
        for (int i = 1; i<M_g;i++){
            for(int j = 1;j<N_g;j++){
                A_g[i+j*(M_g+1)] = A[(i-1)+(j-1)*m];
                B_g[i+j*(M_g+1)] = B[(i-1)+(j-1)*m];
            }
        }
        cudaMalloc((void**)&A_d, (M_g+1) * (N_g+1) * sizeof(float));
        cudaMalloc((void**)&B_d, (M_g+1) * (N_g+1) * sizeof(float));
        cudaMemcpy(A_d, A_g, (M_g+1) * (N_g+1) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_g, (M_g+1) * (N_g+1) * sizeof(float), cudaMemcpyHostToDevice);
        free(A_g); free(B_g);
    }
    b_g = (float*)malloc((M_g+1) * (N_g+1) * sizeof(float));
    Initial= (float*)malloc( (M_g+1) * (N_g+1)* sizeof(float));
    xx_tmp =  (float*)malloc((M_g+1) * (N_g+1) * sizeof(float));

    memset(Initial, 0, (M_g+1) * (N_g+1) * sizeof(float));
    memset(b_g, 0, (M_g+1) * (N_g+1) * sizeof(float));
    cudaMalloc((void**)&rhs_d, (M_g+1) * (N_g+1) * sizeof(float));
    cudaMalloc((void**)&xx1_d, (M_g+1) * (N_g+1) * sizeof(float));
    cudaMalloc((void**)&xx2_d, (M_g+1) * (N_g+1) * sizeof(float));
    cudaMemcpy(xx1_d, Initial, (M_g+1) * (N_g+1) * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(xx2_d, Initial, (M_g+1) * (N_g+1) * sizeof(float),cudaMemcpyHostToDevice);
    
    
    for (int i = 1; i<M_g;i++) for(int j = 1;j<N_g;j++) b_g[i+j*(M_g+1)] = b[(i-1)+(j-1)*(M_g-1)];
    cudaMemcpy(rhs_d, b_g, (M_g+1) * (N_g+1) * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(((M_g+1) + OUT_TILE_SIZE -1) / OUT_TILE_SIZE, ((N_g+1)+OUT_TILE_SIZE -1) / OUT_TILE_SIZE);
    float *tmp;
    int MAXN = 5*M_g*N_g;
    if (tag == 0){
        //fixed 
        for (int i = 0; i<MAXN;i++){
            jacobi_itrs_fixed<<<numBlocks, threadsPerBlock>>>(rhs_d,xx1_d,xx2_d,M_g,N_g,a_c,b_c);
            tmp = xx1_d; xx1_d = xx2_d; xx2_d = tmp;
        }
    }else{
        //not fixed
        for (int i = 0; i<MAXN;i++){
            jacobi_itrs<<<numBlocks, threadsPerBlock>>>(A_d, B_d,rhs_d,xx1_d,xx2_d,M_g,N_g);
            tmp = xx1_d; xx1_d = xx2_d; xx2_d = tmp;
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(xx_tmp, xx2_d, (M_g+1) * (N_g+1) * sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 1;j<N_g;j++)
        for (int i = 1; i<M_g;i++) 
            worker[(i-1)+(j-1)*(M_g-1)] = xx_tmp[i+j*(M_g+1)];
    
    cudaFree(xx1_d); cudaFree(xx2_d); cudaFree(rhs_d);
    if(tag){
        cudaFree(A_d); cudaFree(B_d);
    }
    free(b_g);free(xx_tmp);free(Initial);
}



//Kernel -- one stencil operation, we don't use halo cells
__global__ void stencil(double *xx, double *yy, int m, int n, double aa, double bb){
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    //no halo cells used
    int idx = bx * BLOCK_SIZE + tx ; int idy = by * BLOCK_SIZE + ty;

    __shared__ double xx_s[BLOCK_SIZE][BLOCK_SIZE];

    if ( 0<=idy&& idy < n && 0<=idx&&idx < m){ 
        xx_s[ty][tx] = xx[idx + idy * m];
    }else{
        xx_s[ty][tx] = 0;
    }

    __syncthreads();
    //boundary check
    if (0<=idx && idx < m && 0<=idy && idy < n){
        double tmp = 2*(aa+bb)*xx_s[ty][tx];
        if (ty>0) tmp -= bb*xx_s[ty-1][tx];
        else if(idy>0) tmp -= bb*xx[idx + (idy-1) * m];

        if (ty<BLOCK_SIZE-1) tmp -= bb*xx_s[ty+1][tx];
        else if(idy<n-1) tmp -= bb*xx[idx + (idy+1) * m];

        if (tx>0) tmp -= aa*xx_s[ty][tx-1];
        else if(idx>0) tmp -= aa*xx[idx-1 + idy * m];

        if (tx<BLOCK_SIZE-1) tmp -= aa*xx_s[ty][tx+1];
        else if(idx<m-1) tmp -= aa*xx[idx+1 + idy * m];

        yy[idx + idy * (m)] = tmp;
    }
}


extern "C" void amul(double *xx, double *yy, int M, int N,double a_c,double b_c){
    int n = N - 1;// the actual interior grids height
    int m = M - 1;// the actual interior grids width
    double *xx_d, *yy_d;
    cudaMalloc(&xx_d, m*n * sizeof(double));
    cudaMalloc(&yy_d, m*n * sizeof(double));
    cudaMemcpy(xx_d, xx, m*n * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((m + BLOCK_SIZE -1) / BLOCK_SIZE, (n+BLOCK_SIZE -1) / BLOCK_SIZE);
    stencil<<<numBlocks, threadsPerBlock>>>(xx_d, yy_d, m, n, a_c, b_c);
    cudaDeviceSynchronize();
    cudaMemcpy(yy, yy_d, m*n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(xx_d); cudaFree(yy_d);
}


int GPU_INITED = 0;
cublasHandle_t handle;

extern "C" void cublas_Dgemm(int transa, int transb,
                           int m, int n, int k,
                           double          alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           double          beta,
                           double          *C, int ldc,
                           int A_length,int B_length,int result_length){
    double *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,A_length * sizeof(double));
    cudaMalloc(&d_B,B_length * sizeof(double));
    cudaMalloc(&d_C,result_length * sizeof(double));
 
    cudaMemcpy(d_A,A,A_length * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,B_length * sizeof(double),cudaMemcpyHostToDevice);

    if (GPU_INITED == 0){
        GPU_INITED = 1;
        cublasCreate(&handle);
    }


    //multiplication
    cublasOperation_t trans = CUBLAS_OP_N;
    if (transa){
        trans = CUBLAS_OP_T;
    }// transb is not supported for simplicity, we don't need it in our case

    cublasDgemm(handle, trans, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda,d_B, ldb, &beta, d_C, ldc);
    cudaMemcpy(C,d_C,result_length * sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}