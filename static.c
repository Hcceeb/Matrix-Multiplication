#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define Size 1000   // Matrix size
#define Threads 8
#define Run 10
#define ChunkSize 50
  // Manually set chunk size

void ParallelMatrixMultiplication(int **A, int **B, int **C)
{
    #pragma omp parallel for num_threads(Threads) schedule(static, ChunkSize)
    for (int i = 0; i < Size; i++)
    {
        for (int k = 0; k < Size; k++)  // Optimized loop order
        {
            int temp = A[i][k];  // Reduce memory accesses
            for (int j = 0; j < Size; j++)
            {
                C[i][j] += temp * B[k][j];
            }
        }
    }
}

int main()
{
    // Allocate memory dynamically
    int **A = (int **)malloc(Size * sizeof(int *));
    int **B = (int **)malloc(Size * sizeof(int *));
    int **C = (int **)malloc(Size * sizeof(int *));
    
    for (int i = 0; i < Size; i++)
    {
        A[i] = (int *)malloc(Size * sizeof(int));
        B[i] = (int *)malloc(Size * sizeof(int));
        C[i] = (int *)malloc(Size * sizeof(int));
    }

    // Initialize matrices A and B with random values
    srand(50);
    for (int i = 0; i < Size; i++)
    {
        for (int j = 0; j < Size; j++)
        {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
            C[i][j] = 0; // Initialize C
        }
    }

    double total_time = 0;

    for (int r = 0; r < Run; r++)
    {
        double start = omp_get_wtime();
        ParallelMatrixMultiplication(A, B, C);
        double end = omp_get_wtime();

        double iteration_time = end - start;
        total_time += iteration_time;

        printf("Run %d: Execution Time = %f seconds\n", r + 1, iteration_time);
    }

    double avg_time = total_time / Run;
    printf("Average Execution Time over %d runs: %f seconds\n", Run, avg_time);

    // Free allocated memory
    for (int i = 0; i < Size; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}