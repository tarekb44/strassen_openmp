#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

// Function to allocate memory for a matrix of size n x n
int* allocate_matrix(int n) {
    int* matrix = (int*)malloc(n * n * sizeof(int));
    if (!matrix) {
        printf("Memory allocation failed for size %d x %d.\n", n, n);
        exit(EXIT_FAILURE);
    }
    memset(matrix, 0, n * n * sizeof(int));
    return matrix;
}

// Function to initialize a matrix with random values
void initialize_matrix(int* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() % 10;
    }
}

// Function to print a matrix
void print_matrix(const char* label, int* matrix, int n) {
    printf("%s:\n", label);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

// Function to add two matrices: C = A + B
void add_matrix(int* A, int* B, int* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) {
        C[i] = A[i] + B[i];
    }
}

// Function to subtract two matrices: C = A - B
void subtract_matrix(int* A, int* B, int* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) {
        C[i] = A[i] - B[i];
    }
}

// Function to perform standard matrix multiplication: C = A * B
void standard_multiply(int* A, int* B, int* C, int n) {
    memset(C, 0, n * n * sizeof(int));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Recursive implementation of Strassen's algorithm
void strassen_multiply(int* A, int* B, int* C, int n, int s) {
    if (n <= s) {
        standard_multiply(A, B, C, n);
        return;
    }

    int new_size = n / 2;
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

    // Submatrices for A and B
    for (int i = 0; i < new_size; i++) {
        for (int j = 0; j < new_size; j++) {
            int idx1 = i * n + j;
            int idx2 = i * n + j + new_size;
            int idx3 = (i + new_size) * n + j;
            int idx4 = (i + new_size) * n + j + new_size;

            A11[i * new_size + j] = A[idx1];
            A12[i * new_size + j] = A[idx2];
            A21[i * new_size + j] = A[idx3];
            A22[i * new_size + j] = A[idx4];

            B11[i * new_size + j] = B[idx1];
            B12[i * new_size + j] = B[idx2];
            B21[i * new_size + j] = B[idx3];
            B22[i * new_size + j] = B[idx4];
        }
    }

    // Compute M1 to M7
    int* temp1 = allocate_matrix(new_size);
    int* temp2 = allocate_matrix(new_size);

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

    subtract_matrix(A12, A22, temp1, new_size);
    add_matrix(B21, B22, temp2, new_size);
    strassen_multiply(temp1, temp2, M7, new_size, s);

    // Combine results into C
    for (int i = 0; i < new_size; i++) {
        for (int j = 0; j < new_size; j++) {
            int idx1 = i * n + j;
            int idx2 = i * n + j + new_size;
            int idx3 = (i + new_size) * n + j;
            int idx4 = (i + new_size) * n + j + new_size;

            C[idx1] = M1[i * new_size + j] + M4[i * new_size + j] - M5[i * new_size + j] + M7[i * new_size + j];
            C[idx2] = M3[i * new_size + j] + M5[i * new_size + j];
            C[idx3] = M2[i * new_size + j] + M4[i * new_size + j];
            C[idx4] = M1[i * new_size + j] - M2[i * new_size + j] + M3[i * new_size + j] + M6[i * new_size + j];
        }
    }

    // Free temporary matrices
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(temp1); free(temp2);
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <k> <k_prime> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int k = atoi(argv[1]);
    int k_prime = atoi(argv[2]);
    int num_threads = atoi(argv[3]);

    omp_set_num_threads(num_threads);

    int n = 1 << k;
    int s = 1 << (k - k_prime);

    int* A = allocate_matrix(n);
    int* B = allocate_matrix(n);
    int* C = allocate_matrix(n);
    int* C_standard = allocate_matrix(n);

    initialize_matrix(A, n);
    initialize_matrix(B, n);

    double start_time = omp_get_wtime();
    strassen_multiply(A, B, C, n, s);
    double end_time = omp_get_wtime();
    printf("Strassen's multiplication time: %lf seconds\n", end_time - start_time);

    start_time = omp_get_wtime();
    standard_multiply(A, B, C_standard, n);
    end_time = omp_get_wtime();
    printf("Standard multiplication time: %lf seconds\n", end_time - start_time);

    free(A); free(B); free(C); free(C_standard);
    return 0;
}