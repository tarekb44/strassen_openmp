#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>


#define PRINT 0

int* allocate_matrix(int n);
void print_matrix(const char* label, int* matrix, int n);
void add_matrix(int* A, int* B, int* C, int n);
void subtract_matrix(int* A, int* B, int* C, int n);
void standard_multiply(int* A, int* B, int* C, int n);

//print a matrix
void print_matrix(const char* label, int* matrix, int n)
{
    printf("%s:\n", label);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            printf("%d ", matrix[i * n + j]);
}


void standard_multiply(int* A, int* B, int* C, int n) {
    memset(C, 0, n * n * sizeof(int));


    #pragma omp parallel for collapse(2)
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            for (int k=0; k<n; k++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
}

// Recursive implementation of Strassen's algorithm
void strassen_multiply(int* A, int* B, int* C, int n, int s) {

    // if current matrix size nxn is less than sxs
    //"Terminate the recursion after k’ levels (k > k’ > 1) when you reach matrices of size s x s."
    if (n <= s) 
    {
        standard_multiply(A, B, C, n); // O(n^3)
        return;
    }


    // consider the following partitioning of the matrices into equal-sized blocks: A=[A11 A12; A21 A22], B = [B11 B12; B21 B22]."
    int new_size = n/2;

    //allocate memory for the intermediate M1 - M7
    //[C11 C12; C21 C22] = [M1+M4-M5+M7, M3 + M5; M2 + M4, M1 - M2 + M3 + M6]
    int* M1 = allocate_matrix(new_size);
    int* M2 = allocate_matrix(new_size);
    int* M3 = allocate_matrix(new_size);
    int* M4 = allocate_matrix(new_size);
    int* M5 = allocate_matrix(new_size);
    int* M6 = allocate_matrix(new_size);
    int* M7 = allocate_matrix(new_size);

    int* A11 = allocate_matrix(new_size);
    int* A12 = allocate_matrix(new_size);
    int* A21 = allocate_matrix(new_size);
    int* A22 = allocate_matrix(new_size);

    int* B11 = allocate_matrix(new_size);
    int* B12 = allocate_matrix(new_size);
    int* B21 = allocate_matrix(new_size);
    int* B22 = allocate_matrix(new_size);

    // "Divide the matrices into smaller submatrices of size n/2 x n/2."
    for (int i = 0; i < new_size; i++) {
        for (int j = 0; j < new_size; j++) {
            int idx1 = i * n + j;
            int idx2 = i * n + j + new_size;
            int idx3 = (i+new_size)*n+ j;
            int idx4 = (i+new_size)* n + j + new_size;

            A11[i*new_size+j] = A[idx1];
            A12[i*new_size+j] = A[idx2];
            A21[i*new_size+j] = A[idx3];
            A22[i*new_size+j] = A[idx4];


            B11[i*new_size+j] =B[idx1];
            B12[i*new_size+j] =B[idx2];
            B21[i*new_size+j] =B[idx3];
            B22[i*new_size+j] =B[idx4];
        }
    }


    int* temp1 = allocate_matrix(new_size);
    int* temp2 = allocate_matrix(new_size);
    //"M1 = (A11 + A22)(B11 + B22)."
    add_matrix(A11, A22, temp1, new_size);
    add_matrix(B11, B22, temp2, new_size);
    strassen_multiply(temp1, temp2, M1, new_size, s);
    add_matrix(A21, A22, temp1, new_size);
    strassen_multiply(temp1, B11, M2, new_size, s);
    subtract_matrix(B12, B22, temp2, new_size);
    strassen_multiply(A11, temp2, M3, new_size, s);
    subtract_matrix(B21, B11, temp2, new_size);
    strassen_multiply(A22, temp2, M4, new_size, s);


    add_matrix(A11, A12, temp1, new_size);
    strassen_multiply(temp1, B22, M5, new_size, s);

    subtract_matrix(A21, A11, temp1, new_size);
    add_matrix(B11, B12, temp2, new_size);
    strassen_multiply(temp1, temp2, M6, new_size, s);

    // "M7 = (A12 - A22)(B21 + B22)."
    subtract_matrix(A12, A22, temp1, new_size);
    add_matrix(B21, B22, temp2, new_size);
    strassen_multiply(temp1, temp2, M7, new_size, s);

    // "Combine the intermediate results into C using:
    // [C11, C12; C21, C22] = [M1 + M4 - M5 + M7, M3 + M5; M2 + M4, M1 - M2 + M3 + M6]."
    for (int i = 0; i < new_size; i++)
    {
        for (int j = 0; j < new_size; j++) {
            int idx1 = i*n+j;
            int idx2 = i*n+j+new_size;
            int idx3 = (i+new_size)*n+j;
            int idx4 = (i+new_size)*n+j+new_size;
            C[idx1] = M1[i*new_size+j] + M4[i*new_size+j] - M5[i*new_size+j] + M7[i*new_size+j];
            C[idx2] = M3[i*new_size+j] + M5[i*new_size+j];
            C[idx3] = M2[i*new_size+j] + M4[i*new_size+j];
            C[idx4] = M1[i*new_size+j] - M2[i*new_size+j] + M3[i*new_size+j] + M6[i*new_size + j];
        }
    }

    
    // free them all
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
        free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(temp1); free(temp2);
}

void init_matrix(int* matrix, int n) {
    for (int i = 0; i < n * n; i++)
        matrix[i] = rand() % 100;
}

// Main function
int main(int argc, char* argv[]) {

    if (argc != 4)
    {
        printf("Usage: %s <k> <k_prime> <num_threads>\n", argv[0]);
        exit(1);
    }

    int k = atoi(argv[1]);
    int k_prime = atoi(argv[2]);
    
    // k’ levels (k > k’ > 1)
    if (k_prime >= k || k_prime < 1)
    {
        printf("k_prime must be 1 <= k_prime < k\n");
        return EXIT_FAILURE;
    }

    int num_threads = atoi(argv[3]);

    omp_set_num_threads(num_threads);

    int n = 1 << k; //A and  B  are of size  nxn , n = 2^k  
    int s = 1 << (k - k_prime); // terminate the recursion after  k_prime  levels ( k >k_prime > 1 ) when you reach matrices of size  sxs

    int* A = allocate_matrix(n);
    int* B = allocate_matrix(n);


    // where we store teh strassen multiplication
    int* C = allocate_matrix(n);

    // we will compare the result in this matrix
    int* C_standard = allocate_matrix(n);

    // init with random values
    init_matrix(A, n);
    init_matrix(B, n);

    // measure the time of the strassed multiplcation
    double start_time = omp_get_wtime();
    strassen_multiply(A, B, C, n, s);
    double end_time = omp_get_wtime();
    printf("Strassen's multiplication time: %lf seconds\n", end_time - start_time);

    // measure the time of the standard time
    start_time = omp_get_wtime();
    standard_multiply(A, B, C_standard, n);
    end_time = omp_get_wtime();
    printf("Standard multiplication time: %lf seconds\n", end_time - start_time);


#if PRINT
    print_matrix("matrix A", A, n);
    print_matrix("matrix B", B, n);
#endif

    free(A); free(B); free(C); free(C_standard);


    return 0;
}


//matrix of size n x n
int* allocate_matrix(int n) {
    int* matrix = (int*)malloc(n * n * sizeof(int));

    if (matrix == NULL)
        exit(1);

    memset(matrix, 0, n * n * sizeof(int));

    return matrix;
}

// add two matrixes: C = A + B
void add_matrix(int* A, int* B, int* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++)
        C[i] = A[i]+B[i];
}

//subtract two matrixes: C = A - B
void subtract_matrix(int* A, int* B, int* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++)
        C[i] =A[i]-B[i];
}