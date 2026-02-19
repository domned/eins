// --- EINSUM CODE GENERATOR ---

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

typedef uint32_t IndexBitmap; /* 26 bits used for 'a'..'z' */

// Minimal Matrix type (re-added here so this translation unit is self-contained)
typedef struct {
    double *data;   /* contiguous data buffer */
    int ndim;       /* number of dimensions */
    int *shape;     /* length ndim */
    IndexBitmap indices;
} Matrix;

/* Forward declarations for functions implemented in einsum.c */
IndexBitmap literal_to_bitmap(const char *lit);
void parse_einsum_notation(const char *notation, char *in1, char *in2, char *out);
Matrix* matrix_create_nd(int ndim, const int *shape, const char *indices_str);
void matrix_free(Matrix *m);

/*
 * Helper: Generates the linear index expression for a matrix access.
 * e.g., for indices "ij" and shape [10, 20], generates string: "(i * 20) + j"
 */
void generate_linear_access_string(char *buffer, const char *indices_str, const int *shape, int ndim) {
    if (ndim == 0) {
        strcpy(buffer, "0");
        return;
    }

    buffer[0] = '\0';
    int first_term = 1;

    for (int d = 0; d < ndim; d++) {
        char loop_char = indices_str[d];
        
        // Calculate Stride: The product of all dimensions *after* this one.
        // Row-major: The last dimension has stride 1.
        int stride = 1;
        for (int k = d + 1; k < ndim; k++) {
            stride *= shape[k];
        }

        if (!first_term) strcat(buffer, " + ");
        
        char term[64];
        if (stride == 1) {
            sprintf(term, "%c", loop_char);
        } else {
            sprintf(term, "(%c * %d)", loop_char, stride);
        }
        strcat(buffer, term);
        first_term = 0;
    }
}

/* * Main Generator Function:
 * Analyzes the input tensors and notation, then prints optimized C code.
 */
void generate_einsum_impl(const char *notation, const Matrix *A, const Matrix *B) {
    char in1[32], in2[32], out[32];
    
    // Reuse your existing parser
    parse_einsum_notation(notation, in1, in2, out); //

    // 1. Identify which letters correspond to which dimensions in A and B
    // We map 'a' -> 0, 'b' -> 1, ... up to 'z' -> 25
    int dim_sizes[26]; 
    for(int i=0; i<26; i++) dim_sizes[i] = -1;

    // Fill dimension sizes based on Matrix A
    for (int i = 0; i < A->ndim; i++) {
        char c = in1[i];
        if (c >= 'a' && c <= 'z') dim_sizes[c - 'a'] = A->shape[i]; //
    }
    // Fill dimension sizes based on Matrix B (validation check could go here)
    for (int i = 0; i < B->ndim; i++) {
        char c = in2[i];
        if (c >= 'a' && c <= 'z') {
            int sz = B->shape[i]; //
            // In a real compiler, we would check if dim_sizes[c-'a'] matches sz
            if (dim_sizes[c - 'a'] == -1) dim_sizes[c - 'a'] = sz;
        }
    }

    // 2. Determine Loop Order
    // Standard optimization: Outer loops = Output indices. Inner loops = Summation indices.
    char loop_order[32];
    int loop_count = 0;

    // Add Output indices first (Outer loops)
    for (int i = 0; out[i]; i++) {
        loop_order[loop_count++] = out[i];
    }

    // Add Summation indices (Indices in Inputs but NOT in Output)
    IndexBitmap bm_out = literal_to_bitmap(out); //
    IndexBitmap bm_in = literal_to_bitmap(in1) | literal_to_bitmap(in2); //
    
    for (int c = 0; c < 26; c++) {
        // If char is in input bitmap AND NOT in output bitmap
        if ((bm_in & (1u << c)) && !(bm_out & (1u << c))) {
            loop_order[loop_count++] = (char)('a' + c);
        }
    }
    loop_order[loop_count] = '\0';


    // 3. CODE GENERATION PHASE
    printf("// ========================================================\n");
    printf("// Auto-generated implementation for einsum(\"%s\")\n", notation);
    printf("// Shapes: A{"); 
    for(int i=0; i<A->ndim; i++) printf("%d%s", A->shape[i], i==A->ndim-1?"":",");
    printf("}, B{");
    for(int i=0; i<B->ndim; i++) printf("%d%s", B->shape[i], i==B->ndim-1?"":",");
    printf("}\n");
    printf("// ========================================================\n\n");

    printf("void einsum_generated(const double *A, const double *B, double *C) {\n");

    // Print Nested Loops
    for (int k = 0; k < loop_count; k++) {
        char loop_char = loop_order[k];
        int size = dim_sizes[loop_char - 'a'];
        
        // Print indentation based on depth
        for (int s = 0; s < k + 1; s++) printf("    ");
        
        printf("for (int %c = 0; %c < %d; ++%c) {\n", loop_char, loop_char, size, loop_char);
    }

    // 4. Generate Math Core
    char access_A[128];
    char access_B[128];
    char access_C[128];
    
    // We need the shape of C to calculate strides for C. 
    // We construct it dynamically based on the 'out' string and our known dim_sizes.
    int shape_C[32];
    int ndim_C = strlen(out);
    for(int i=0; i<ndim_C; i++) shape_C[i] = dim_sizes[out[i] - 'a'];

    // Generate the index math strings (e.g., "i * 3 + j")
    generate_linear_access_string(access_A, in1, A->shape, A->ndim);
    generate_linear_access_string(access_B, in2, B->shape, B->ndim);
    generate_linear_access_string(access_C, out, shape_C, ndim_C);

    // Print the accumulation line
    for (int s = 0; s < loop_count + 1; s++) printf("    "); // Indent
    printf("    C[%s] += A[%s] * B[%s];\n", access_C, access_A, access_B);

    // Close Loops
    for (int k = loop_count - 1; k >= 0; k--) {
        for (int s = 0; s < k + 1; s++) printf("    ");
        printf("}\n");
    }
    printf("}\n");
}


int main(void) {
    printf("//--- Generating Code ---\n");

    int shapeA[2] = {7, 3};
    int shapeB[2] = {3, 6};
    Matrix *A = matrix_create_nd(2, shapeA, "ij");
    Matrix *B = matrix_create_nd(2, shapeB, "jc");
    if (!A || !B) {
        fprintf(stderr, "Failed to allocate sample matrices\n");
        matrix_free(A);
        matrix_free(B);
        return 1;
    }

    generate_einsum_impl("ij,jc->ic", A, B);

    matrix_free(A);
    matrix_free(B);
    return 0;
}