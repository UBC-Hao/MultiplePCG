#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>
#include <sys/resource.h>
#include "hw4_lib.h"

#define BLOCK_SIZE 32
#define OUT_TILE_SIZE ((BLOCK_SIZE) - 2)
#define A_s(i,j) A_s[j][i]
#define B_s(i,j) B_s[j][i]

namespace cg = cooperative_groups;

__global__ void jacobi(float *rhs, float *xx1, float *xx2, int M, int N, float aa, float bb,int itrs) {
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  auto g = cg::this_grid();

  // we use halo cells
  // these are the real indices of this thread
  int idx = bx * OUT_TILE_SIZE + tx - 1; int idy = by * OUT_TILE_SIZE + ty - 1;

  __shared__ float rhs_s[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float xx1_s[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float xx2_s[BLOCK_SIZE][BLOCK_SIZE];
  // The indexing is for linearly indexing the 2D array
  if ( 0<=idy&& idy < N+1 && 0<=idx&&idx < M+1){ //boundary check
      rhs_s[ty][tx] = rhs[idx + idy * (M+1)];
      xx1_s[ty][tx] = xx1[idx + idy * (M+1)];
  }else{
      rhs_s[ty][tx] = 0;
      xx1_s[ty][tx] = 0;
  }
  xx2_s[ty][tx] = 0;
  
  float *tmp;
  float tmp2;
  float *out = xx2;

  __syncthreads();
  for (int k = 0;k<itrs;k++){
    //boundary check
    // 5-point stencil operation
    if (0<idx && idx < M && 0<idy && idy < N & tx<BLOCK_SIZE-1 & ty<BLOCK_SIZE-1 & tx>0 & ty>0){
        xx2_s[ty][tx]= (rhs_s[ty][tx] + bb*(xx1_s[ty+1][tx] + xx1_s[ty-1][tx]) + aa*(xx1_s[ty][tx+1] + xx1_s[ty][tx-1]))/(2*(aa+bb));
        //xx2_s[ty][tx]= (xx1_s[ty][tx]+xx1_s[ty+1][tx] + xx1_s[ty-1][tx] + xx1_s[ty][tx+1] + xx1_s[ty][tx-1])/5.0;
  
    }

    tmp = xx1; xx1 = xx2; xx2 = tmp;
    if (tx == 1 || ty == 1 || tx == BLOCK_SIZE - 2 || ty == BLOCK_SIZE - 2){
      if (0<idx && idx < M && 0<idy && idy < N & tx<BLOCK_SIZE-1 & ty<BLOCK_SIZE-1 & tx>0 & ty>0){
        //we use xx1 and xx2 as a way to communicate between blocks,
        // in odd iterations, we read and write from the original xx2,
        // and in even iterations, we read and write from the original xx1
        xx1[idx + idy*(M+1)] = xx2_s[ty][tx];
      }
    }

    g.sync();//sync iter-blocks

    //swap xx1_s and xx2_s
    tmp2 = xx1_s[ty][tx]; xx1_s[ty][tx] = xx2_s[ty][tx]; xx2_s[ty][tx] = tmp2; 

    //update halo cells
    if (tx == 0 || ty == 0|| tx == BLOCK_SIZE - 1 || ty == BLOCK_SIZE - 1){
      if ( 0<=idy&& idy < N+1 && 0<=idx&&idx < M+1){
        xx1_s[ty][tx] = xx1[idx + idy*(M+1)];
      }
    }
    __syncthreads();
  }

  
  if (0<idx && idx < M && 0<idy && idy < N & tx<BLOCK_SIZE-1 & ty<BLOCK_SIZE-1 & tx>0 & ty>0){
      out[idx + idy*(M+1)] = xx1_s[ty][tx];
  }
}

int main(int argc, char **argv) {
  const char *opt_names[] = {"stencil", "N", NULL};
  uint N = get_uint(argc, argv, opt_names, 1, 50, 1);
  // problem size
  int itrs =  5*N*N;
  size_t bytes = (N+1)*(N+1) * sizeof(float);
  float *xx1, *xx2, *b;
  float aa = 1.0, bb = 1.0;
  cudaMallocManaged(&xx1, bytes);// we use this malloc for simplicity
  cudaMallocManaged(&xx2, bytes);
  cudaMallocManaged(&b, bytes);

  for (int i = 1; i<N;i++) for(int j = 1;j<N;j++) b[i+j*(N+1)] = (1.0/N)*(1.0/N);
  for (int i = 0; i<N+1;i++) for(int j = 0;j<N+1;j++) {xx1[i+j*(N+1)] = 0; xx2[i+j*(N+1)] = 0;}

  
  void *kernelArgs[] = { &b,&xx1,&xx2,&N,&N,&aa,&bb,&itrs};
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(((N+1) + OUT_TILE_SIZE -1) / OUT_TILE_SIZE, ((N+1)+OUT_TILE_SIZE -1) / OUT_TILE_SIZE);
  struct rusage r0, r1; // for timing measurements
  getrusage(RUSAGE_SELF, &r0);
  cudaLaunchCooperativeKernel((void*)jacobi, numBlocks, threadsPerBlock, kernelArgs);
  cudaDeviceSynchronize();
  getrusage(RUSAGE_SELF, &r1); 
  double gpu_time =   (r1.ru_utime.tv_sec - r0.ru_utime.tv_sec)
		     + 1e-6*(r1.ru_utime.tv_usec - r0.ru_utime.tv_usec);
  printf("GPU time: %f seconds\n", gpu_time);
  float max = -10;
  for (int i = 0; i<N+1;i++) for(int j = 0;j<N+1;j++) if (xx2[i+j*(N+1)]>max){max = xx2[i+j*(N+1)];}
  printf("max: %f\n",max);
  return 0;
}