/*THIS IS TEST FOR GPU IMPLEMENTAION OF SINGLE-PRECISION MAT-MAT MULTIPICATION
  BY JIARUI LU 03/2020
  USING CUDA, initial version without optimization
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>



#define THREAD_NUM 256 // the number of threads in each block
#define ROW_LENGTH // the row of the matrix : m
#define EPSILON 0.01 // the error limit for identification

void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("\n");
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %ld.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("\n");
}


//initialization of the left matrix and the right one
void mat_init(float *mat,int row, int col)
{
    int i,j;
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            // random single-precision(4) value assignment
            mat[(i*col) + j] = (float)rand() / (float)RAND_MAX + ( (i+j)%2==1?1:-1 );
            //mat[(i*col) + j] = 1.0;
        }
    }
}
// gpu kernel of multiplication
__global__ void mat_mul(float *lmat, float *rmat, float *res_mat, int m, int n)
{

    // get the id of block and thread id
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // EACH THREAD takes charge of 1 row of res_mat 
    const int row = bid * THREAD_NUM + tid;
    int j,k;
    float sum_tmp;
    // EACH THREAD takes charge of 1 row of res_mat, there are 
    for(j=0;j<m;j++)
        {
            sum_tmp = 0.0;
            for(k=0;k<n;k++) // n is common 'border' of l and r
                sum_tmp += lmat[ (row*n) + k] * rmat[ (k*n) + j]; //cache-resue bad for rmat
            // random single-precision(4) value assignment
            res_mat[(row*m) + j] = sum_tmp;
        }
    
}
    

int main(int argc, char *argv[])
{
    // print the device information
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, 0);
    printDeviceProp(dprop);
    /* declaration of matrix and their size: 
    lmat: mxn ; rmat: nxm ; res_mat: mxm 
    using normal linear array representing the matrix:
    A[i,j] = arr[(i-1)*n + j]*/
    int m, n;
    if(argc==3)
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }
    if(m%THREAD_NUM!=0)//input checking
    {
        printf("Error: not input the right size of m/n! (m=%d, n=%d) \n",m,n);
        exit(1);
    }
    printf("Param accepted as: m = %d; n= %d \n", m, n);
    float *lmat, *rmat, *res_mat;
    float *g_lmat, *g_rmat, *g_res_mat;
    // allocation of the memory
    lmat    = (float *)malloc(sizeof(float) * m * n);
    rmat    = (float *)malloc(sizeof(float) * n * m);
    res_mat = (float *)malloc(sizeof(float) * m * m);
    //random seed setting
    srand((unsigned) time(NULL));
    //init
    mat_init(lmat, m, n);
    mat_init(rmat, m, n);
    printf("Matrix initialization done.\n");
    
    // a timer for computation
    struct timeval start, end;
    double elapse_t;
    
    //printf("This is a test sentence[%d]\n",test_count++);
    // declare the gpu memory space
    cudaMalloc((void **)&g_lmat,    sizeof(float) * m * n);
    cudaMalloc((void **)&g_rmat,    sizeof(float) * n * m);
    cudaMalloc((void **)&g_res_mat, sizeof(float) * m * m);
    
    gettimeofday(&start,NULL);
    //copy data to v-memory
    cudaMemcpy(g_lmat,lmat, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(g_rmat,rmat, sizeof(float) * n * m, cudaMemcpyHostToDevice);
    //printf("This is a test sentence[%d]\n",test_count++);
    
    // exec the computation
    mat_mul <<< m/THREAD_NUM, THREAD_NUM >>> (g_lmat,g_rmat, g_res_mat ,m ,n);
    
    //copy data back to main memory
    cudaMemcpy(res_mat, g_res_mat, sizeof(float) * m * m, cudaMemcpyDeviceToHost);
    gettimeofday(&end,NULL);
    elapse_t = (end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec)/1000000.0; // in sec
    
    // free cuda pointer
    cudaFree(g_lmat);
    cudaFree(g_rmat);
    cudaFree(g_res_mat);

    
    printf("###### Identification #######\n");
    int i,j,k;

    int flag=0;
    

    //CPU version of the multiplication
    // each area of the res_mat is independent -> so truly parallism is allowed
    float sum_tmp;
    for(i=0;i<m;i++)
    {
        for(j=0;j<m;j++)
        {
            sum_tmp = 0.0;
            for(k=0;k<n;k++)
                sum_tmp += lmat[ (i*n) + k] * rmat[ (k*n) + j]; //cache-resue bad for rmat
            // random single-precision(4) value assignment
            if( (res_mat[(i*m) + j] - sum_tmp)> EPSILON )
                flag += 1;
        }
    }

    if(flag==0)
        printf("Identification Passed\n");
    else
        printf("Identification Failed: flag = %d\n", flag);
    

    printf("\n#########################################\n\n");
    printf("Finished, elapsed %.3lf seconds\n", elapse_t);
    printf("Real performance: %.3lf GFlops\n", (float)m * (float)m * (2.0 * (float)n)/( elapse_t * 1000000000.0));
    printf("\n#########################################\n");
    return 0;

}