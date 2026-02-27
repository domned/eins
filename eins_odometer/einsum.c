#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../einsum.h"

// Convert literal like "ij" to bitmap (only lowercase a..z)
IndexBitmap literal_to_bitmap(const char *lit) {
    IndexBitmap bm = 0;
    if (!lit) return 0;
    for (const char *p = lit; *p; p++) 
    {
        char c = *p;
        if (c >= 'a' && c <= 'z')  
        {
            bm |= (1u << (c - 'a')); //perform bitwise OR(left shift == 2^<index>, so we normalize into a single 1)

        }
    }
    return bm;
}

// Print the indices contained in a bitmap 
void print_bitmap_indices(IndexBitmap bm) {
    int first = 1;
    putchar('{');
    for (int i = 0; i < 26; i++) {
        if (bm & (1u << i)) {
            if (!first) putchar(',');
            putchar('a' + i);
            first = 0;
        }
    }
    putchar('}');
}

// Print bitmap as binary (26 bits: a..z -> bit 0..25). Leftmost is 'z'.
void print_bitmap_binary(IndexBitmap bm) {
    putchar('0'); putchar('b');
    for (int i = 25; i >= 0; --i) {
        putchar((bm & (1u << i)) ? '1' : '0');
        if (i % 4 == 0 && i != 0) putchar('_'); // group nibble-like for readability
    }
}

// Create an N-D matrix from an array of dimension sizes
Matrix* matrix_create_nd(int ndim, const int *shape, const char *indices_str) {
    if (ndim <= 0 || !shape) return NULL;
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->ndim = ndim;
    m->shape = (int*)malloc(sizeof(int) * ndim);
    if (!m->shape) { free(m); return NULL; }
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        m->shape[i] = shape[i];
    if (shape[i] <= 0) { free(m->shape); free(m); return NULL; }
    }
    for (int i = 0; i < ndim; i++) total *= (size_t)m->shape[i];
    m->indices = literal_to_bitmap(indices_str);
    m->data = (double*)calloc(total, sizeof(double));
    if (!m->data) { free(m->shape); free(m); return NULL; }
    return m;
}


void matrix_free(Matrix *m) {
    if (!m) return;
    free(m->data);
    free(m->shape);
    free(m);
}

// Compute linear offset from indices (array of length ndim)
static size_t compute_offset(const Matrix *m, const int *idx) {
    // Compute row-major linear offset
    // offset = (((idx[0]*shape[1] + idx[1])*shape[2] + idx[2]) ... )
    size_t off = 0;
    for (int d = 0; d < m->ndim; d++) {
        off = off * (size_t)m->shape[d] + (size_t)idx[d];
    }
    return off;
}

// N-D set/get
void matrix_set_nd(Matrix *m, const int *idx, double v) {
    if (!m || !idx) return;
    for (int d = 0; d < m->ndim; d++) {
        if (idx[d] < 0 || idx[d] >= m->shape[d]) return;
    }
    size_t off = compute_offset(m, idx);
    m->data[off] = v;
}

double matrix_get_nd(const Matrix *m, const int *idx) {
    if (!m || !idx) return 0.0;
    for (int d = 0; d < m->ndim; d++) {
        if (idx[d] < 0 || idx[d] >= m->shape[d]) return 0.0;
    }
    size_t off = compute_offset(m, idx);
    return m->data[off];
}

double matrix_get(const Matrix *m, int i, int j) {
    if (!m || m->ndim != 2) return 0.0;
    int idx[2] = { i, j };
    return matrix_get_nd(m, idx);
}

void matrix_set(Matrix *m, int i, int j, double v) {
    if (!m || m->ndim != 2) return;
    int idx[2] = { i, j };
    matrix_set_nd(m, idx, v);
}

// Advance the multi-dimensional index vector by one step.
// idx: current index vector of length `length` (row-major order: idx[0] is most-significant)
// shape: per-dimension sizes of length `length`
// finished: pointer to int flag; set to 1 when no further combinations exist
// This increments the index vector in lexicographic (row-major) order where the
// last element is the least-significant digit (fastest-changing).
void next_indices(int *idx, const int *shape, int length, int *finished) {
    if (!idx || !shape || length <= 0) {
        if (finished) *finished = 1;
        return;
    }

    int j = length - 1; // start with least-significant dimension
    // carry while the current dimension cannot advance
    while (j >= 0 && (idx[j] + 1) >= shape[j]) {
        idx[j] = 0; // reset (carry)
        j--;
    }

    if (j < 0) {
        // we've overflowed the most-significant digit: iteration finished
        if (finished) *finished = 1;
        return;
    }

    // advance the first dimension that can still increment
    idx[j]++;
    if (finished) *finished = 0;
}

void matrix_print(const Matrix *m) {
    if (!m) { printf("(null)\n"); return; }
    if (m->ndim == 2) {
        printf("Matrix %dx%d indices=", m->shape[0], m->shape[1]);
    } else {
        printf("Matrix (ndim=%d) shape=({", m->ndim);
        for (int d = 0; d < m->ndim; d++) {
            if (d) printf(",");
            printf("%d", m->shape[d]);
        }
        printf("}) indices=");
    }
    print_bitmap_indices(m->indices);
    printf("\n");
    if (m->ndim == 2) {
        for (int r = 0; r < m->shape[0]; ++r) {
            for (int c = 0; c < m->shape[1]; ++c) {
                printf("%8.4f ", matrix_get(m, r, c));
            }
            printf("\n");
        }
    } else {
        // print flat listing for ND
        size_t total = 1;
        for (int d = 0; d < m->ndim; d++) total *= (size_t)m->shape[d];
        for (size_t i = 0; i < total; i++) {
            printf("%8.4f ", m->data[i]);
            if ((i + 1) % 8 == 0) printf("\n");
        }
        printf("\n");
    }
}


void parse_einsum_notation(const char *notation, char *in1, char *in2, char *out) {
    // Find arrow
    const char *arrow = strstr(notation, "->");
    if (arrow) {
        strcpy(out, arrow + 2);
    } else {
        out[0] = '\0';
    }

    // Copy left side into temp and split on comma
    size_t left_len = arrow ? (size_t)(arrow - notation) : strlen(notation);
    char left[128];
    if (left_len >= sizeof(left)) left_len = sizeof(left) - 1;
    memcpy(left, notation, left_len);
    left[left_len] = '\0';
    char *comma = strchr(left, ',');
    if (comma) {
        *comma = '\0';
        strcpy(in1, left);
        strcpy(in2, comma + 1);
    } else {
        strcpy(in1, left);
        in2[0] = '\0';
    }
}

Matrix* einsum(const char *notation, const Matrix *A, const Matrix *B){
    if (!notation || !A || !B) return NULL;
    char in1[32], in2[32], out[32];
    parse_einsum_notation(notation, in1, in2, out);

    // bitmaps for quick membership tests
    IndexBitmap bm1 = literal_to_bitmap(in1);
    IndexBitmap bm2 = literal_to_bitmap(in2);
    IndexBitmap bmout = literal_to_bitmap(out);

    // Build position maps: for each letter, which axis in A/B/out (-1 if not present)
    int posA[26], posB[26], posOut[26];
    for (int i = 0; i < 26; i++) { posA[i] = posB[i] = posOut[i] = -1; }
    for (int i = 0; in1[i]; i++) {
        char c = in1[i];
        if (c >= 'a' && c <= 'z') posA[c - 'a'] = i;
    }
    for (int i = 0; in2[i]; i++) {
        char c = in2[i];
        if (c >= 'a' && c <= 'z') posB[c - 'a'] = i;
    }
    for (int i = 0; out[i]; i++) {
        char c = out[i];
        if (c >= 'a' && c <= 'z') posOut[c - 'a'] = i;
    }

    // Validate that the literal length matches matrix ndim for A and B
    if ((int)strlen(in1) != A->ndim) return NULL;
    if ((int)strlen(in2) != B->ndim) return NULL;

    // Determine summed indices: indices present in inputs but not in output
    IndexBitmap input_mask = bm1 | bm2;
    IndexBitmap sum_mask = input_mask & (~bmout);

    // Build list of combined indices: first output indices, then summed indices
    char combined[52];
    int combined_len = 0;
    // add output letters in order
    for (int i = 0; out[i]; i++) combined[combined_len++] = out[i];
    // add summed letters (in alphabetical order) that are not in output
    for (int c = 0; c < 26; c++) {
        if ( (sum_mask & (1u << c)) ) {
            // avoid adding letters already present in output
            if (posOut[c] == -1) combined[combined_len++] = (char)('a' + c);
        }
    }

    // Build output shape from out string
    int out_ndim = (int)strlen(out);
    int *out_shape = NULL;
    if (out_ndim > 0) {
        out_shape = (int*)malloc(sizeof(int) * out_ndim);
        if (!out_shape) return NULL;
        for (int i = 0; i < out_ndim; i++) {
            char c = out[i];
            int ci = c - 'a';
            int dim = -1;
            if (posA[ci] >= 0) dim = A->shape[posA[ci]];
            if (posB[ci] >= 0) {
                int dimb = B->shape[posB[ci]];
                if (dim == -1) dim = dimb;
                else if (dim != dimb) { free(out_shape); return NULL; }
            }
            if (dim <= 0) { free(out_shape); return NULL; }
            out_shape[i] = dim;
        }
    }

    Matrix *C = matrix_create_nd(out_ndim > 0 ? out_ndim : 1, out_shape ? out_shape : (int[]){1}, out);
    if (out_shape) free(out_shape);
    if (!C) return NULL;

    // Combined iteration shape: for each combined letter determine size
    int comb_len = combined_len;
    int *comb_shape = (int*)malloc(sizeof(int) * comb_len);
    if (!comb_shape) { matrix_free(C); return NULL; }
    for (int k = 0; k < comb_len; k++) {
        char c = combined[k];
        int ci = c - 'a';
        int dim = -1;
        if (posA[ci] >= 0) dim = A->shape[posA[ci]];
        if (posB[ci] >= 0) {
            int dimb = B->shape[posB[ci]];
            if (dim == -1) dim = dimb;
            else if (dim != dimb) { free(comb_shape); matrix_free(C); return NULL; }
        }
        if (dim <= 0) { free(comb_shape); matrix_free(C); return NULL; }
        comb_shape[k] = dim;
    }

    // iterate over combined indices using next_indices
    int *comb_idx = (int*)malloc(sizeof(int) * comb_len);
    int *idxA = (int*)malloc(sizeof(int) * A->ndim);
    int *idxB = (int*)malloc(sizeof(int) * B->ndim);
    if (!comb_idx || !idxA || !idxB) { free(comb_idx); free(idxA); free(idxB); free(comb_shape); matrix_free(C); return NULL; }
    for (int k = 0; k < comb_len; k++) comb_idx[k] = 0;
    for (int a = 0; a < A->ndim; a++) idxA[a] = 0;
    for (int b = 0; b < B->ndim; b++) idxB[b] = 0;

    int finished = 0;
    // We'll iterate all combinations; for each, write to output coordinate (first out_ndim entries)
    do {
        // map combined indices to A and B coords
        for (int a = 0; a < A->ndim; a++) idxA[a] = 0;
        for (int b = 0; b < B->ndim; b++) idxB[b] = 0;
        for (int k = 0; k < comb_len; k++) {
            char c = combined[k];
            int ci = c - 'a';
            int val = comb_idx[k];
            if (posA[ci] >= 0) idxA[posA[ci]] = val;
            if (posB[ci] >= 0) idxB[posB[ci]] = val;
        }

        // compute where to accumulate in C (output indices are first portion)
        int out_idx_len = out_ndim > 0 ? out_ndim : 1;
        int *out_idx = (int*)malloc(sizeof(int) * out_idx_len);
        if (!out_idx) break;
        for (int i = 0; i < out_idx_len; i++) {
            // find the position of out[i] in combined
            char c = out[i];
            int pos = -1;
            for (int k = 0; k < comb_len; k++) if (combined[k] == c) { pos = k; break; }
            out_idx[i] = (pos >= 0) ? comb_idx[pos] : 0;
        }

        // Multiply A and B values for this combined position and add to C at out_idx
        double a_val = matrix_get_nd(A, idxA);
        double b_val = matrix_get_nd(B, idxB);
        double prod = a_val * b_val;

        // read current C value and add
        double cur = matrix_get_nd(C, out_idx);
        matrix_set_nd(C, out_idx, cur + prod);
        free(out_idx);

        next_indices(comb_idx, comb_shape, comb_len, &finished);
    } while (!finished);

    free(comb_idx);
    free(idxA);
    free(idxB);
    free(comb_shape);
    return C;
}






#ifndef EINSUM_NO_MAIN
int main(void) {
    const char *notation = "ij,jc->ic";
    char in1[32], in2[32], out[32];

    printf("Testing parser and bitmap helpers\n");
    parse_einsum_notation(notation, in1, in2, out);
    printf("Parsed: in1='%s' in2='%s' out='%s'\n", in1, in2, out);

    IndexBitmap bm1 = literal_to_bitmap(in1);
    IndexBitmap bm2 = literal_to_bitmap(in2);
    printf("Bitmap in1: "); print_bitmap_indices(bm1); printf(" (0x%x, ", bm1); print_bitmap_binary(bm1); printf(")\n");
    printf("Bitmap in2: "); print_bitmap_indices(bm2); printf(" (0x%x, ", bm2); print_bitmap_binary(bm2); printf(")\n");
    printf("Common:    "); print_bitmap_indices(bm1 & bm2); printf(" (0x%x, ", bm1 & bm2); print_bitmap_binary(bm1 & bm2); printf(")\n");

    int shapeA[2] = {2, 3};
    int shapeB[2] = {3, 2};
    Matrix *A = matrix_create_nd(2, shapeA, in1);
    Matrix *B = matrix_create_nd(2, shapeB, in2);
    if (!A || !B) {
        fprintf(stderr, "Failed to allocate matrices\n");
        matrix_free(A);
        matrix_free(B);
        return 1;
    }

    /* Fill A: 1..6, B: 101..106 */
    for (int i = 0; i < A->shape[0]; ++i)
        for (int j = 0; j < A->shape[1]; ++j)
            matrix_set(A, i, j, (double)(i * A->shape[1] + j + 1));

    for (int i = 0; i < B->shape[0]; ++i)
        for (int j = 0; j < B->shape[1]; ++j)
            matrix_set(B, i, j, (double)(100 + i * B->shape[1] + j + 1));

    /* Iterate A using next_indices and print each coordinate + value */
    {
        int ndim = A->ndim;
        int *idx = (int*)malloc(sizeof(int) * ndim);
        if (idx) {
            for (int d = 0; d < ndim; ++d) idx[d] = 0;
            int finished = 0;
            printf("\nIterating A with next_indices:\n");
            do {
                printf("(");
                for (int d = 0; d < ndim; ++d) {
                    if (d) printf(",");
                    printf("%d", idx[d]);
                }
                double v = matrix_get_nd(A, idx);
                printf(") = %g\n", v);
                next_indices(idx, A->shape, ndim, &finished);
            } while (!finished);
            free(idx);
        }
    }

    printf("\nMatrix A:\n"); matrix_print(A);
    printf("\nMatrix B:\n"); matrix_print(B);
    // Test einsum: compute C = einsum("ij,jc->ic", A, B)
    Matrix *C = einsum("ij,jc->ic", A, B);
    if (C) {
        printf("\nResult C (einsum ij,jc->ic):\n");
        matrix_print(C);
        matrix_free(C);
    } else {
        fprintf(stderr, "einsum returned NULL\n");
    }

    matrix_free(A);
    matrix_free(B);
    return 0;
}
#endif // EINSUM_NO_MAIN








/*
https://github.com/l0r3m1psum/GOFAIitsabouttime/blob/a511ca16464fefcfa73074b4194f15fb9960988a/algorithm.cpp#L855C1-L873C1
https://en.wikipedia.org/wiki/Row-_and_column-major_order#Address_calculation_in_general
https://en.wikipedia.org/wiki/Mixed_radix
*/