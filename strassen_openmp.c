// strassen_openmp_enhanced.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>



// Function to free allocated matrix memory
void free_matrix(int* matrix) {
    free(matrix);
}

// Function to copy a submatrix from src to dest
void copy_submatrix(int* src, int* dest, int src_row, int src_col, int n, int sub_size) {
    for(int i = 0; i < sub_size; i++) {
        memcpy(dest + i * sub_size, src + (src_row + i) * n + src_col, sub_size * sizeof(int));
    }
}

// Function to add two matrices: C = A + B
void add_matrix(int* A, int* B, int* C, int n) {
    int i;
    #pragma omp parallel for private(i)
    for(i = 0; i < n * n; i++) {
        C[i] = A[i] + B[i];
    }
}

// Function to subtract two matrices: C = A - B
void subtract_matrix(int* A, int* B, int* C, int n) {
    int i;
    #pragma omp parallel for private(i)
    for(i = 0; i < n * n; i++) {
        C[i] = A[i] - B[i];
    }
}

// Function for standard matrix multiplication: C = A * B
void standard_multiply(int* A, int* B, int* C, int n) {
    int i, j, k;
    #pragma omp parallel for private(j, k) collapse(2)
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            int sum = 0;
            for(k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// allocate matric of size n x n
int* allocate_matrix(int n)
{
    int *matrix = (int*)malloc(n*n*sizeof(int));

    if (matrix == NULL)
        exit(EXIT_FAILURE);
   
   memset(matrix, 0, n*n * sizeof(int));

    return matrix;
}

// Recursive Strassen's multiplication: C = A * B
void strassen_multiply(int* A, int* B, int* C, int n, int s, int depth) {
    // Logging the current recursive depth and matrix size
    //printf("Recursion Depth: %d | Matrix Size: %d x %d\n", depth, n, n);

    if (n <= s) {
        // printf("Base case reached at depth %d with matrix size %d x %d.\n", depth, n, n);
        standard_multiply(A, B, C, n);
        return;
    }

    int new_size = n / 2;

    // Allocate memory for submatrices of A and B
    int *A11 = allocate_matrix(new_size);
    int *A12 = allocate_matrix(new_size);
    int *A21 = allocate_matrix(new_size);
    int *A22 = allocate_matrix(new_size);

    int *B11 = allocate_matrix(new_size);
    int *B12 = allocate_matrix(new_size);
    int *B21 = allocate_matrix(new_size);
    int *B22 = allocate_matrix(new_size);

    // Copy data into submatrices
    copy_submatrix(A, A11, 0, 0, n, new_size);
    copy_submatrix(A, A12, 0, new_size, n, new_size);
    copy_submatrix(A, A21, new_size, 0, n, new_size);
    copy_submatrix(A, A22, new_size, new_size, n, new_size);

    copy_submatrix(B, B11, 0, 0, n, new_size);
    copy_submatrix(B, B12, 0, new_size, n, new_size);
    copy_submatrix(B, B21, new_size, 0, n, new_size);
    copy_submatrix(B, B22, new_size, new_size, n, new_size);

    // Allocate memory for M1 to M7
    int *M1 = allocate_matrix(new_size);
    int *M2 = allocate_matrix(new_size);
    int *M3 = allocate_matrix(new_size);
    int *M4 = allocate_matrix(new_size);
    int *M5 = allocate_matrix(new_size);
    int *M6 = allocate_matrix(new_size);
    int *M7 = allocate_matrix(new_size);

    // Compute M1 = (A11 + A22) * (B11 + B22)
    #pragma omp task shared(M1, A11, A22, B11, B22)
    {
        int* tempA = allocate_matrix(new_size);
        int* tempB = allocate_matrix(new_size);
        add_matrix(A11, A22, tempA, new_size);
        add_matrix(B11, B22, tempB, new_size);
        strassen_multiply(tempA, tempB, M1, new_size, s, depth + 1);
        free_matrix(tempA);
        free_matrix(tempB);
    }

    // Compute M2 = (A21 + A22) * B11
    #pragma omp task shared(M2, A21, A22, B11)
    {
        int* tempA = allocate_matrix(new_size);
        add_matrix(A21, A22, tempA, new_size);
        strassen_multiply(tempA, B11, M2, new_size, s, depth + 1);
        free_matrix(tempA);
    }

    // Compute M3 = A11 * (B12 - B22)
    #pragma omp task shared(M3, A11, B12, B22)
    {
        int* tempB = allocate_matrix(new_size);
        subtract_matrix(B12, B22, tempB, new_size);
        strassen_multiply(A11, tempB, M3, new_size, s, depth + 1);
        free_matrix(tempB);
    }

    // Compute M4 = A22 * (B21 - B11)
    #pragma omp task shared(M4, A22, B21, B11)
    {
        int* tempB = allocate_matrix(new_size);
        subtract_matrix(B21, B11, tempB, new_size);
        strassen_multiply(A22, tempB, M4, new_size, s, depth + 1);
        free_matrix(tempB);
    }

    // Compute M5 = (A11 + A12) * B22
    #pragma omp task shared(M5, A11, A12, B22)
    {
        int* tempA = allocate_matrix(new_size);
        add_matrix(A11, A12, tempA, new_size);
        strassen_multiply(tempA, B22, M5, new_size, s, depth + 1);
        free_matrix(tempA);
    }

    // Compute M6 = (A21 - A11) * (B11 + B12)
    #pragma omp task shared(M6, A21, A11, B11, B12)
    {
        int* tempA = allocate_matrix(new_size);
        int* tempB = allocate_matrix(new_size);
        subtract_matrix(A21, A11, tempA, new_size);
        add_matrix(B11, B12, tempB, new_size);
        strassen_multiply(tempA, tempB, M6, new_size, s, depth + 1);
        free_matrix(tempA);
        free_matrix(tempB);
    }

    // Compute M7 = (A12 - A22) * (B21 + B22)
    #pragma omp task shared(M7, A12, A22, B21, B22)
    {
        int* tempA = allocate_matrix(new_size);
        int* tempB = allocate_matrix(new_size);
        subtract_matrix(A12, A22, tempA, new_size);
        add_matrix(B21, B22, tempB, new_size);
        strassen_multiply(tempA, tempB, M7, new_size, s, depth + 1);
        free_matrix(tempA);
        free_matrix(tempB);
    }

    // Wait for all tasks to complete
    #pragma omp taskwait

    // Allocate memory for C11, C12, C21, C22
    int *C11 = allocate_matrix(new_size);
    int *C12 = allocate_matrix(new_size);
    int *C21 = allocate_matrix(new_size);
    int *C22 = allocate_matrix(new_size);

    // Compute C11 = M1 + M4 - M5 + M7
    add_matrix(M1, M4, C11, new_size);
    subtract_matrix(C11, M5, C11, new_size);
    add_matrix(C11, M7, C11, new_size);

    // Compute C12 = M3 + M5
    add_matrix(M3, M5, C12, new_size);

    // Compute C21 = M2 + M4
    add_matrix(M2, M4, C21, new_size);

    // Compute C22 = M1 - M2 + M3 + M6
    subtract_matrix(M1, M2, C22, new_size);
    add_matrix(C22, M3, C22, new_size);
    add_matrix(C22, M6, C22, new_size);

    // Combine C11, C12, C21, C22 into C
    for(int i = 0; i < new_size; i++) {
        memcpy(C + i * n, C11 + i * new_size, new_size * sizeof(int)); // C11
        memcpy(C + i * n + new_size, C12 + i * new_size, new_size * sizeof(int)); // C12
        memcpy(C + (i + new_size) * n, C21 + i * new_size, new_size * sizeof(int)); // C21
        memcpy(C + (i + new_size) * n + new_size, C22 + i * new_size, new_size * sizeof(int)); // C22
    }

    // Logging the combination step
    // printf("Combined submatrices at depth %d to form the final matrix.\n", depth);

    // Free allocated memory for submatrices and temporaries
    free_matrix(A11);
    free_matrix(A12);
    free_matrix(A21);
    free_matrix(A22);

    free_matrix(B11);
    free_matrix(B12);
    free_matrix(B21);
    free_matrix(B22);

    free_matrix(M1);
    free_matrix(M2);
    free_matrix(M3);
    free_matrix(M4);
    free_matrix(M5);
    free_matrix(M6);
    free_matrix(M7);

    free_matrix(C11);
    free_matrix(C12);
    free_matrix(C21);
    free_matrix(C22);
}

// Function to initialize a matrix with random integers
void initialize_matrix(int* matrix, int n) {
    int i;

    for(i = 0; i < n * n; i++)
        matrix[i] = rand() % 10;
}

// Function to print a matrix
void print_matrix(int* matrix, int n) {
    int i, j;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

// Function to verify if two matrices are equal
int verify_result(int* C1, int* C2, int n) {
    int i;
    for(i = 0; i < n * n; i++) {
        if(C1[i] != C2[i]) {
            return 0; // Not equal
        }
    }
    return 1; // Equal
}

int main(int argc, char* argv[]) {
    if(argc != 4)
    {
        printf("Usage: %s <k> <k_prime> <num_threads>\n", argv[0]);
        printf("Where:\n");
        printf("  n = 2^k (Matrix size)\n");
        printf("  s = 2^(k - k_prime) (Termination size)\n");
        printf("  num_threads = Number of OpenMP threads to use\n");
        return EXIT_FAILURE;
    }

    int k = atoi(argv[1]);
    int k_prime = atoi(argv[2]);
    int num_threads = atoi(argv[3]);

    if(k_prime >= k || k_prime < 1) {
        printf("invalid k_prime\n");
        return EXIT_FAILURE;
    }

    if(num_threads < 1)
    {
        printf("threads must be min 1\n");
        return EXIT_FAILURE;
    }

    omp_set_num_threads(num_threads);

    printf("Number of threads set to: %d\n", num_threads);
    
    int n = 1 << k; // 2^k
    printf("Matrix size (n x n): %d x %d\n", n, n);

    int s = 1 << (k - k_prime); // 2^k-k_prime
    printf("Termination size (s x s): %d x %d\n", s, s);

    // Allocate and initialize matrices
    int* A = allocate_matrix(n); // matrix A = first input matrix
    int* B = allocate_matrix(n); // matrix B = second input matrix
    int* C = allocate_matrix(n); // store the product of A and B from strassen method


    int* C_standard = allocate_matrix(n); //allocate memory for standard multiplcations (for comparison)

    // start generating array of size 
    srand(time(NULL));

    printf("init matrices A and B with random integers...\n");
    initialize_matrix(A, n);
    initialize_matrix(B, n);

    printf("matrix a:\n");
    print_matrix(A, n);
    printf("maatrix B:\n");
    print_matrix(B, n);

    // Start the measurment of time of the strassen algorithm
    double start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            strassen_multiply(A, B, C, n, s, 1);
        }
    }
    double end_time = omp_get_wtime();
    // ewnd the time and record it

    printf("strassen's multiplication time: %lf secs \n", end_time - start_time);

    //start time for the standard multiplication
    start_time = omp_get_wtime();
    standard_multiply(A, B, C_standard, n);
    // end time and record it for standard
    end_time = omp_get_wtime();

    printf("standard multiplication time: %lf secs\n", end_time - start_time);

    // Verify the result
    printf("Verifying the results...\n");
    if(verify_result(C, C_standard, n)) {
        printf("Verification: SUCCESS. Strassen's result matches standard multiplication.\n");
    }
    else {
        printf("Verification: FAILURE. Results do not match.\n");

        // Debugging: Print matrices for small n
        if(n <= 8) { // Adjust threshold as needed
            printf("Matrix A:\n");
            print_matrix(A, n);
            printf("Matrix B:\n");
            print_matrix(B, n);
            printf("Matrix C (Strassen):\n");
            print_matrix(C, n);
            printf("Matrix C_standard:\n");
            print_matrix(C_standard, n);
        }
    }

    // Free allocated memory
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(C_standard);

    return EXIT_SUCCESS;
}
