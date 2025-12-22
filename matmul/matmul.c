#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../einsum.h"

// Creates a new matrix with dimensions permuted according to `order`
// e.g. order = {1, 0} swaps rows and columns (transpose)
Matrix* matrix_permute(const Matrix *src, const int *order) {
    if (!src || !order) return NULL;
    
    // 1. Create the new shape
    int *new_shape = (int*)malloc(sizeof(int) * src->ndim);
    for (int i = 0; i < src->ndim; i++) {
        new_shape[i] = src->shape[order[i]];
    }
    
    // We cannot construct the "indices" bitmap easily here without the original string, 
    // but for the math logic, we just need shape and data. 
    // Passing "per" as dummy string.
    Matrix *dst = matrix_create_nd(src->ndim, new_shape, "per");
    free(new_shape);

    // 2. Copy data to new layout
    // We iterate through the DESTINATION linearly to keep writes contiguous (cache friendly),
    // and calculate the SOURCE index for each read.
    size_t total_elements = 1;
    for (int i = 0; i < src->ndim; i++) total_elements *= src->shape[i];

    int *coords = (int*)calloc(src->ndim, sizeof(int)); // coordinates in DST
    int *src_coords = (int*)malloc(sizeof(int) * src->ndim);

    for (size_t i = 0; i < total_elements; i++) {
        // Map dst coordinates to src coordinates using the inverse of `order`
        // If dst dim [0] came from src dim [order[0]], then src coordinate at [order[0]] is coords[0]
        for (int d = 0; d < src->ndim; d++) {
            src_coords[order[d]] = coords[d];
        }

        double val = matrix_get_nd(src, src_coords); //
        dst->data[i] = val; // Linear write

        // Increment odometer for dst (coords)
        for (int d = src->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < dst->shape[d]) break;
            coords[d] = 0;
        }
    }

    free(coords);
    free(src_coords);
    return dst;
}


Matrix* einsum_matmul(const char *notation, const Matrix *A, const Matrix *B) {
    char in1[32], in2[32], out[32];
    parse_einsum_notation(notation, in1, in2, out); //

    // --- STEP 1: ANALYZE INDICES ---
    // Identify Free indices (Keep) vs Summation indices (Contract)
    IndexBitmap bm1 = literal_to_bitmap(in1);
    IndexBitmap bm2 = literal_to_bitmap(in2);
    IndexBitmap bmOut = literal_to_bitmap(out);
    
    // Summation indices are those in (Input1 U Input2) - Output
    IndexBitmap bmSum = (bm1 | bm2) & (~bmOut);

    // Lists to store the permutation orders
    int permA[10], permB[10]; 
    int countA_free = 0, countA_sum = 0;
    int countB_free = 0, countB_sum = 0;
    
    // Build Permutation for A: [Free indices..., Summation indices...]
    // This groups all free dims to the left, and sum dims to the right.
    int pA_idx = 0;
    // 1a. Add A's free indices
    for (int i = 0; i < A->ndim; i++) {
        char c = in1[i];
        if (bmOut & (1u << (c-'a'))) permA[pA_idx++] = i;
    }
    countA_free = pA_idx;
    // 1b. Add A's summation indices
    for (int i = 0; i < A->ndim; i++) {
        char c = in1[i];
        if (bmSum & (1u << (c-'a'))) permA[pA_idx++] = i;
    }
    countA_sum = pA_idx - countA_free;

    // Build Permutation for B: [Summation indices..., Free indices...]
    // Note order: Sum indices on Left (to align with A's Right columns)
    int pB_idx = 0;
    // 2a. Add B's summation indices
    // CRITICAL: Must be in same relative order as they appear in A's permuted end
    // To match A's sum group, we must look at how we ordered A's sum group.
    // However, simplest is to just sort alphabetically or order by appearance. 
    // Let's iterate A's sum indices and find where they are in B.
    for (int k = countA_free; k < A->ndim; k++) {
        // The char at A's k-th permuted dimension
        char target = in1[permA[k]];
        // Find this char in B
        char *ptr = strchr(in2, target);
        if (ptr) permB[pB_idx++] = (int)(ptr - in2);
    }
    countB_sum = pB_idx;
    
    // 2b. Add B's free indices
    for (int i = 0; i < B->ndim; i++) {
        char c = in2[i];
        if (bmOut & (1u << (c-'a'))) permB[pB_idx++] = i;
    }
    countB_free = pB_idx - countB_sum;

    // --- STEP 2: PERMUTE TENSORS ---
    Matrix *permutedA = matrix_permute(A, permA);
    Matrix *permutedB = matrix_permute(B, permB);

    // --- STEP 3: CALCULATE FLATTENED DIMENSIONS ---
    int rows_A = 1; 
    for(int i=0; i<countA_free; i++) rows_A *= permutedA->shape[i];
    
    int cols_A_sum = 1;
    for(int i=countA_free; i<permutedA->ndim; i++) cols_A_sum *= permutedA->shape[i];
    
    int rows_B_sum = 1; // Should equal cols_A_sum
    for(int i=0; i<countB_sum; i++) rows_B_sum *= permutedB->shape[i];
    
    int cols_B = 1;
    for(int i=countB_sum; i<permutedB->ndim; i++) cols_B *= permutedB->shape[i];

    if (cols_A_sum != rows_B_sum) {
        fprintf(stderr, "Dimension mismatch in contraction: %d vs %d\n", cols_A_sum, rows_B_sum);
        return NULL;
    }

    // --- STEP 4: MATRIX MULTIPLICATION (rows_A x cols_A_sum) * (rows_B_sum x cols_B) ---
    // C_flat will be (rows_A x cols_B)
    double *C_data = (double*)calloc(rows_A * cols_B, sizeof(double));
    
    // Standard GEMM (Naive implementation for demo - Replace with BLAS/OpenBLAS)
    for (int i = 0; i < rows_A; i++) {
        for (int k = 0; k < cols_A_sum; k++) {
            double a_val = permutedA->data[i * cols_A_sum + k];
            for (int j = 0; j < cols_B; j++) {
                double b_val = permutedB->data[k * cols_B + j];
                C_data[i * cols_B + j] += a_val * b_val;
            }
        }
    }

    // Cleanup permuted temporaries
    matrix_free(permutedA);
    matrix_free(permutedB);

    // --- STEP 5: RESHAPE/PERMUTE OUTPUT ---
    // Currently C_data is laid out as [A_Free_Dims..., B_Free_Dims...]
    // We need to reshape it to that N-dim shape, then permute it to match 'out' string.
    
    // 5a. Construct the shape of the intermediate result
    int intermediate_ndim = countA_free + countB_free;
    int *intermediate_shape = (int*)malloc(sizeof(int) * intermediate_ndim);
    char intermediate_indices[32]; 
    int idx_ptr = 0;
    
    // Reconstruct the indices string for the intermediate result
    for (int i=0; i<countA_free; i++) {
        char original_char = in1[permA[i]];
        intermediate_indices[idx_ptr] = original_char;
        // Need to find the size of this char. It is in permutedA, dim i.
        // Actually we destroyed permutedA. We must look up original A.
        // permA[i] is the original index in A.
        intermediate_shape[idx_ptr] = A->shape[permA[i]];
        idx_ptr++;
    }
    for (int i=0; i<countB_free; i++) {
        // The free indices of B were at the END of permB
        int original_dim_idx = permB[countB_sum + i]; 
        intermediate_indices[idx_ptr] = in2[original_dim_idx];
        intermediate_shape[idx_ptr] = B->shape[original_dim_idx];
        idx_ptr++;
    }
    intermediate_indices[idx_ptr] = '\0';

    Matrix *C_intermediate = NULL;
    if (intermediate_ndim <= 0) {
        // Scalar intermediate: represent as a 1-element 1-D matrix
        int one = 1;
        C_intermediate = matrix_create_nd(1, &one, "s");
        if (!C_intermediate) {
            free(intermediate_shape);
            free(C_data);
            return NULL;
        }
        free(C_intermediate->data);
        C_intermediate->data = C_data; // take ownership
        free(intermediate_shape);

        // If output is also scalar (out length 0), return directly
        if (out[0] == '\0') {
            return C_intermediate;
        }
        // Otherwise, fall through: we'll attempt to permute a 1-D container
    } else {
        C_intermediate = matrix_create_nd(intermediate_ndim, intermediate_shape, intermediate_indices);
        if (!C_intermediate) {
            free(intermediate_shape);
            free(C_data);
            return NULL;
        }
        free(C_intermediate->data); // replace calloc'd data with our computed data
        C_intermediate->data = C_data; // take ownership
        free(intermediate_shape);
    }

    // 5b. Final Permutation to match requested 'out' string
    // We have C_intermediate with indices e.g. "ij" but user might want "ji".
    int final_perm[16];
    for (int i = 0; out[i]; i++) {
        char target = out[i];
        char *ptr = strchr(intermediate_indices, target);
        if (ptr) {
            final_perm[i] = (int)(ptr - intermediate_indices);
        } else {
            // if target not found, default to 0
            final_perm[i] = 0;
        }
    }

    // If intermediate_ndim was 0 we created a 1-D container; ensure permutation length matches
    if (intermediate_ndim <= 0) {
        Matrix *FinalResult = matrix_permute(C_intermediate, final_perm);
        matrix_free(C_intermediate);
        return FinalResult;
    }

    Matrix *FinalResult = matrix_permute(C_intermediate, final_perm);
    matrix_free(C_intermediate);
    return FinalResult;
}