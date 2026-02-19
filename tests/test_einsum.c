#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../einsum.h"

// If USE_GENERATED is defined at compile-time, the test will use the generated
// implementation `einsum_generated(const double*, const double*, double*)`.
// Otherwise it will call the native `einsum()` function.

#ifdef USE_GENERATED
// Expect `gen_impl.c` to be provided at build time or present in repo root.
extern void einsum_generated(const double *A, const double *B, double *C);
#endif

// If MATMUL_IMPL is defined, use the matmul implementation symbol
#ifdef MATMUL_IMPL
extern Matrix* einsum_matmul(const char *notation, const Matrix *A, const Matrix *B);
#endif

static void fill_A(double *A, int r, int c) {
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) A[i*c + j] = (double)(i*c + j + 1);
}
static void fill_B(double *B, int r, int c) {
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) B[i*c + j] = (double)(100 + i*c + j + 1);
}

static void naive_einsum_ij_jc__ic(const double *A, int Ar, int Ac,
                                   const double *B, int Br, int Bc,
                                   double *out /* size Ar x Bc */) {
    // Compute C[i,c] = sum_j A[i,j] * B[j,c]
    for (int i = 0; i < Ar; ++i) {
        for (int c = 0; c < Bc; ++c) {
            double s = 0.0;
            for (int j = 0; j < Ac; ++j) s += A[i*Ac + j] * B[j*Bc + c];
            out[i*Bc + c] = s;
        }
    }
}

static int compare_arrays(const double *a, const double *b, size_t n) {
    const double eps = 1e-9;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs(a[i] - b[i]);
        if (!(d <= eps || d / (fabs(b[i]) + 1e-12) <= 1e-6)) return 0;
    }
    return 1;
}

int main(void) {
    // Define multiple test cases (notation, shapes)
    struct TestCase {
        const char *notation;
        int A_ndim; int A_shape[4];
        int B_ndim; int B_shape[4];
    } tests[] = {
        { "ij,jc->ic", 2, {7,3,0,0}, 2, {3,6,0,0} },   // matmul (matches generated impl)
        { "i,i->",     1, {10,0,0,0}, 1, {10,0,0,0} }, // dot product (vectors length 10)
        { "i,j->ij",   1, {5,0,0,0},  1, {4,0,0,0} },   // outer product (5 x 4)
        { "ab,bc->ac", 2, {4,3,0,0},  2, {3,6,0,0} },   // matmul variant
        { "abc,cd->abd",3, {2,3,3,0}, 2, {3,5,0,0} }    // tensor contraction: A(2,3,3) * B(3,5) -> C(2,3,5)
    };

    const int ntests = sizeof(tests)/sizeof(tests[0]);
    int total_failed = 0;

    for (int ti = 0; ti < ntests; ++ti) {
        const char *notation = tests[ti].notation;
        int Ar = tests[ti].A_ndim;
        int Br = tests[ti].B_ndim;
        int A_nd = tests[ti].A_ndim;
        int B_nd = tests[ti].B_ndim;
        int Ashape[4], Bshape[4];
        for (int i=0;i<4;i++) { Ashape[i] = tests[ti].A_shape[i]; Bshape[i] = tests[ti].B_shape[i]; }
        int Cr = (notation[0] == '\0') ? 1 : Ar; // placeholder, will compute properly below

        // compute C dims by parsing output string
        char in1[32], in2[32], out[32];
        parse_einsum_notation(notation, in1, in2, out);
        int out_nd = (int)strlen(out);
        int shapeC[4] = {1,1,1,1};
        // Map letters to sizes
        // Map letters to sizes using the explicit shapes provided
        int sizes[26]; for (int i=0;i<26;i++) sizes[i]=-1;
        // fill from A's indices string
        for (int d = 0; d < A_nd; ++d) if (in1[d]>='a' && in1[d]<='z') sizes[in1[d]-'a'] = Ashape[d];
        for (int d = 0; d < B_nd; ++d) if (in2[d]>='a' && in2[d]<='z') sizes[in2[d]-'a'] = Bshape[d];
        // Fallback: compute C size as product of output letters mapped from inputs
        int C_elements = 1;
        for (int o = 0; o < out_nd; ++o) {
            char c = out[o];
            int dim = sizes[c - 'a'];
            if (dim <= 0) dim = 1;
            shapeC[o] = dim;
            C_elements *= dim;
        }

        // compute flattened sizes for A and B from shapes
        size_t szA = 1; for (int d=0; d<A_nd; ++d) szA *= (size_t)Ashape[d];
        size_t szB = 1; for (int d=0; d<B_nd; ++d) szB *= (size_t)Bshape[d];
        size_t szC = (size_t)1;
        for (int o = 0; o < out_nd; ++o) szC *= shapeC[o];
        if (out_nd == 0) szC = 1; // scalar stored as single element

        double *A = (double*)malloc(sizeof(double) * szA);
        double *B = (double*)malloc(sizeof(double) * szB);
        double *C_impl = (double*)calloc(szC, sizeof(double));
        double *C_ref = (double*)malloc(sizeof(double) * szC);
        if (!A || !B || !C_impl || !C_ref) { fprintf(stderr, "allocation failure\n"); return 2; }

        // fill inputs with simple patterns (row-major)
        for (size_t i=0;i<szA;i++) A[i] = (double)(i+1);
        for (size_t i=0;i<szB;i++) B[i] = (double)(100 + i + 1);

        // Print test header
        printf("\n=== Test %d/%d: %s ===\n", ti+1, ntests, notation);
        printf("Impl: ");
    #ifdef USE_GENERATED
        printf("generated\n");
    #elif defined(MATMUL_IMPL)
        printf("matmul (einsum_matmul)\n");
    #else
        printf("native (einsum)\n");
    #endif
        printf("A shape: ("); for (int d=0; d<A_nd; ++d) printf("%d%s", Ashape[d], d==A_nd-1?"":","); printf(")\n");
        printf("B shape: ("); for (int d=0; d<B_nd; ++d) printf("%d%s", Bshape[d], d==B_nd-1?"":","); printf(")\n");

        // Compute reference depending on notation
        if (strcmp(notation, "ij,jc->ic") == 0 || strcmp(notation, "ab,bc->ac") == 0) {
            // matrix multiply: A (Ar x Ac) * B (Br x Bc) where Ac == Br
            int A_r = Ashape[0], A_c = Ashape[1];
            int B_r = Bshape[0], B_c = Bshape[1];
            for (int i = 0; i < A_r; ++i) for (int c = 0; c < B_c; ++c) {
                double s = 0.0;
                for (int j = 0; j < A_c; ++j) s += A[i*A_c + j] * B[j*B_c + c];
                C_ref[i*B_c + c] = s;
            }
        } else if (strcmp(notation, "i,i->") == 0) {
            double s = 0.0;
            for (int i = 0; i < Ashape[0]; ++i) s += A[i] * B[i];
            C_ref[0] = s;
        } else if (strcmp(notation, "i,j->ij") == 0) {
            for (int i = 0; i < Ashape[0]; ++i) for (int j = 0; j < Bshape[0]; ++j) C_ref[i*Bshape[0] + j] = A[i] * B[j];
        } else if (strcmp(notation, "abc,cd->abd") == 0) {
            // A shape (a,b,c) flattened as A[(i*b + j)*c + k]
            int a = Ashape[0], b = Ashape[1], c = Ashape[2], d = Bshape[1];
            for (int i = 0; i < a; ++i) for (int j = 0; j < b; ++j) for (int kk = 0; kk < d; ++kk) {
                double s = 0.0;
                for (int k = 0; k < c; ++k) s += A[(i*b + j)*c + k] * B[k*d + kk];
                C_ref[(i*b + j)*d + kk] = s;
            }
        } else {
            // unknown notation: skip
            printf("SKIP unknown notation %s\n", notation);
            free(A); free(B); free(C_impl); free(C_ref);
            continue;
        }

        // Run implementation under test
#ifdef USE_GENERATED
        // Only run generated impl for the case that matches its dims (7x3 * 3x6)
        if (Ar == 7 && Ac == 3 && Br == 3 && Bc == 6) {
            einsum_generated(A, B, C_impl);
        } else {
            printf("SKIP generated impl for notation %s (shape mismatch)\n", notation);
            free(A); free(B); free(C_impl); free(C_ref);
            continue;
        }
#else
        // Native: use Matrix API and einsum()
        Matrix *mA = matrix_create_nd(A_nd, Ashape, in1);
        Matrix *mB = matrix_create_nd(B_nd, Bshape, in2);
        if (!mA || !mB) { fprintf(stderr, "matrix_create_nd failed\n"); return 2; }
        // copy flat buffers into matrix storage (same row-major layout)
        memcpy(mA->data, A, sizeof(double) * szA);
        memcpy(mB->data, B, sizeof(double) * szB);

        Matrix *mC;
    #ifdef MATMUL_IMPL
        mC = einsum_matmul(notation, mA, mB);
    #else
        mC = einsum(notation, mA, mB);
    #endif
        if (!mC) { fprintf(stderr, "einsum returned NULL for %s\n", notation); return 2; }
        // copy mC data into flat C_impl buffer
        memcpy(C_impl, mC->data, sizeof(double) * szC);

        matrix_free(mA); matrix_free(mB); matrix_free(mC);
#endif

        int ok = compare_arrays(C_impl, C_ref, szC);
        if (ok) {
            printf("Result: PASS\n");
        } else {
            printf("Result: FAIL\n");
            printf("Notation: %s\n", notation);
            printf("Expected (%zu elems):\n", szC);
            for (size_t idx = 0; idx < szC; ++idx) printf("%10.4f ", C_ref[idx]);
            printf("\nGot (%zu elems):\n", szC);
            for (size_t idx = 0; idx < szC; ++idx) printf("%10.4f ", C_impl[idx]);
            printf("\n");
            total_failed++;
        }

        free(A); free(B); free(C_impl); free(C_ref);
    }

    if (total_failed == 0) printf("ALL TESTS PASSED\n");
    else printf("%d TEST(S) FAILED\n", total_failed);
    return total_failed == 0 ? 0 : 1;
}
