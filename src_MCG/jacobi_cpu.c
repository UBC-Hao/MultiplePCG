#include <stdlib.h>
#include <string.h>

#define A(i,j) A[(i)+(j)*m]
#define B(i,j) B[(i)+(j)*m]
#define x1(i,j) x1[(i)+(j)*m]
#define x2(i,j) x2[(i)+(j)*m]
#define b(i,j) b[(i)+(j)*m]

// Jacobi solver using CPU, will implement this in GPU
// TODO: If have time, we will also consider other stationary point methods 
void subdomain_solve_cpu(int m, int n, double *A, double *B, double *b,double *x2){
    double *x1 = (double *)malloc(m*n*sizeof(double));
    double *tmpx1 = x1;
    double *tmp;
    memcpy(x1,b,m*n*sizeof(double));
    for (int k = 0;k<5*m*n;k++){
    for (int j = 0;j<n;j++){
        for (int i = 0;i<m;i++){
            double aa = A(i,j);
            double bb = B(i,j);
            double center = 2*(aa+bb);
            double up = 0, down= 0, left= 0, right=0;
            if (j<n-1) up = -bb*x1(i,j+1);
            if (j>0) down = -bb*x1(i,j-1);
            if (i>0) left = -aa*x1(i-1,j);
            if (i<m-1) right = -aa*x1(i+1,j);
            x2(i,j) = (b(i,j)-(up+down+left+right))/center;
            
        }
    }
    //swap x1 and x2
     tmp = x1; x1 = x2; x2 = tmp;
    }
    free(tmpx1);
}
