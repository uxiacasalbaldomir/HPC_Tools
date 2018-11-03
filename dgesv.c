#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mkl_lapacke.h"


double *generate_matrix(int size)
{
    int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);
    srand(1);

    for (i = 0; i < size * size; i++)
    {
        matrix[i] = rand() % 100;
    }

    return matrix;
}

void print_matrix(const char *name, double *matrix, int size)
{
    int i, j;
    printf("matrix: %s \n", matrix);

    for (i = 0; i < size; i++)
    {
            for (j = 0; j < size; j++)
            {
                printf("%f ", matrix[i * size + j]);
            }
            printf("\n");
    }
}

int check_result(double *bref, double *b, int size) {
    int i;
    float error = 0;
    for(i=0;i<size*size;i++) {
       error = bref[i]-b[i];
    }
    if (error>0.05) return 0;
    else return 1;
}

double *my_dgesv(int n, int nrhs, double *A, double *B) {
	
	double *L = (double*)calloc(n * n, sizeof(double));
    double *U = (double*)calloc(n * n, sizeof(double));
	double *Y = (double*)calloc(n * nrhs, sizeof(double));
    double *X = (double*)calloc(n * nrhs, sizeof(double));

    double res;

	if ((U==NULL) || (L==NULL) || (Y==NULL) || (X==NULL)){
		exit(EXIT_FAILURE);
	}
	int i, j, k;

    #pragma omp parallel private(i,j,k)
    {
        #pragma omp for reduction(-:res)
        for (i = 0; i <n; i++) {
            for (j = 0; j < n; j++){
        	   if(i<=j){
            	   res=A[i*n+j];
                	for(k=0;k<=(i-1);k++){
                		res-=L[i*n+k]*U[k*n+j];
                 	}
                 	U[i*n+j] = res;
                	if (i==j)
                   		L[i*n+j]=1;
           		   else
               		   L[i*n+j]=0;
            	}else{
            		res=A[i*n+j];
           	    	for(k=0; k<=j-1; k++)
               	    	res-=L[i*n+k]*U[k*n+j];
           		   L[i*n+j]=res/U[j*n+j];
           		   U[i*n+j]=0;
        	   }
            }
        }
   
		//LY = B  
        #pragma omp for reduction(-:res)
		for (k=0; k<n;k++){
            for(i=0;i<nrhs;i++){
				res = B[i*nrhs + k];
				for (j=0; j<i;j++){
					res -= L[i*n + j]*Y[j*nrhs + k]; //
				}
				Y[i*nrhs + k] = res;
			}
		}
		
	//Ux = Y
        #pragma omp for reduction(-:res)
        for (k = 0 ; k< nrhs; k++){
            for(i=(n-1); i>=0; i--){
                res= Y[i * nrhs +k];
				for(j=i+1; j<n; j++)
	               res-=U[i*n+j]*X[j*n+k];
				X[i*n+k]=res/U[i*n+i];
			}
        }
	}

	free(L);
	free(U);
	free(Y);
	return X;
}


    void main(int argc, char *argv[])
    {

        int size = atoi(argv[1]);

        double *a, *aref;
        double *b, *bref;

        a = generate_matrix(size);
        aref = generate_matrix(size);        
        b = generate_matrix(size);
        bref = generate_matrix(size);

        //print_matrix("A", a, size);
        //print_matrix("B", b, size);

        // Using MKL to solve the system
        MKL_INT n = size, nrhs = size, lda = size, ldb = size, info;
        MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);

        clock_t tStart = clock();
        info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
        printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

        double start = omp_get_wtime();   
        double *res =my_dgesv(n, nrhs, a, b);
        printf("Time taken by my implementation: %.2fs\n", (omp_get_wtime()-start));
        
        if (check_result(bref,res,size)==1)
            printf("Result is ok!\n");
        else    
            printf("Result is wrong!\n");
        
        //print_matrix("X", bref, size);
        //print_matrix("Xref", res, size);
        free(res);
        free(bref);
        free(b);
        free(aref);
        free(a);
        free(ipiv);
    }
