
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ===== TYPE DEFINITIONS ===== */

typedef uint32_t IndexBitmap;

typedef struct {
    double *data;
    int ndim;
    int *shape;
    IndexBitmap indices;
} Matrix;

typedef enum {
    BLAS_DOT,      /* scalar dot product */
    BLAS_AXPY,     /* vector y := y + alpha*x */
    BLAS_GER,      /* outer product */
    BLAS_GEMV,     /* y := A*x or x := A^T*y */
    BLAS_GEMM,     /* C := A*B */
    BLAS_GENERAL   /* fallback: general tensor contraction */
} BLASPattern;

typedef struct {
    BLASPattern pattern;
    int has_output;
    int num_sum_indices;
    int num_free_indices;
} BLASAnalysis;


IndexBitmap literal_to_bitmap(const char *lit) {
    IndexBitmap bm = 0;
    if (!lit) return 0;
    for (const char *p = lit; *p; p++) {
        char c = *p;
        if (c >= 'a' && c <= 'z') {
            bm |= (1u << (c - 'a'));
        }
    }
    return bm;
}

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

void print_bitmap_binary(IndexBitmap bm) {
    putchar('0'); putchar('b');
    for (int i = 25; i >= 0; --i) {
        putchar((bm & (1u << i)) ? '1' : '0');
        if (i % 4 == 0 && i != 0) putchar('_');
    }
}

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
        total *= (size_t)m->shape[i];
    }
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

static size_t compute_offset(const Matrix *m, const int *idx) {
    size_t off = 0;
    for (int d = 0; d < m->ndim; d++) {
        off = off * (size_t)m->shape[d] + (size_t)idx[d];
    }
    return off;
}

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

void next_indices(int *idx, const int *shape, int length, int *finished) {
    if (!idx || !shape || length <= 0) {
        if (finished) *finished = 1;
        return;
    }
    int j = length - 1;
    while (j >= 0 && (idx[j] + 1) >= shape[j]) {
        idx[j] = 0;
        j--;
    }
    if (j < 0) {
        if (finished) *finished = 1;
        return;
    }
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
    const char *arrow = strstr(notation, "->");
    if (arrow) {
        strcpy(out, arrow + 2);
    } else {
        out[0] = '\0';
    }
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


static BLASAnalysis analyze_blas_pattern(const char *notation, 
                                         const Matrix *A, 
                                         const Matrix *B) {
    BLASAnalysis result = {BLAS_GENERAL, 0, 0, 0};
    
    char in1[32], in2[32], out[32];
    parse_einsum_notation(notation, in1, in2, out);
    
    int ndim_A = A->ndim;
    int ndim_B = B->ndim;
    int ndim_out = (int)strlen(out);
    
    IndexBitmap bm1 = literal_to_bitmap(in1);
    IndexBitmap bm2 = literal_to_bitmap(in2);
    IndexBitmap bm_out = literal_to_bitmap(out);
    
    IndexBitmap input_mask = bm1 | bm2;
    IndexBitmap sum_mask = input_mask & (~bm_out);
    
    int num_sum_indices = 0;
    for (int i = 0; i < 26; i++) {
        if (sum_mask & (1u << i)) num_sum_indices++;
    }
    result.num_sum_indices = num_sum_indices;
    result.num_free_indices = ndim_out;
    result.has_output = (ndim_out > 0) ? 1 : 0;
    
    /* DOT: "i,i->" */
    if (ndim_A == 1 && ndim_B == 1 && ndim_out == 0 && 
        in1[0] == in2[0] && in1[0] >= 'a' && in1[0] <= 'z') {
        result.pattern = BLAS_DOT;
        return result;
    }
    
    /* AXPY: "i,i->i" */
    if (ndim_A == 1 && ndim_B == 1 && ndim_out == 1 && 
        in1[0] == in2[0] && in1[0] == out[0]) {
        result.pattern = BLAS_AXPY;
        return result;
    }
    
    /* GER: "i,j->ij" */
    if (ndim_A == 1 && ndim_B == 1 && ndim_out == 2 &&
        ((in1[0] == out[0] && in2[0] == out[1]) ||
         (in1[0] == out[1] && in2[0] == out[0]))) {
        result.pattern = BLAS_GER;
        return result;
    }
    
    /* GEMV: "ij,j->i" */
    if (ndim_A == 2 && ndim_B == 1 && ndim_out == 1 &&
        in1[1] == in2[0] && in1[0] == out[0]) {
        result.pattern = BLAS_GEMV;
        return result;
    }
    
    /* GEMV: "i,ij->j" */
    if (ndim_A == 1 && ndim_B == 2 && ndim_out == 1 &&
        in1[0] == in2[0] && in2[1] == out[0]) {
        result.pattern = BLAS_GEMV;
        return result;
    }
    
    /* GEMM: "ij,jk->ik" */
    if (ndim_A == 2 && ndim_B == 2 && ndim_out == 2 &&
        in1[1] == in2[0] && in1[0] == out[0] && in2[1] == out[1]) {
        result.pattern = BLAS_GEMM;
        return result;
    }
    
    result.pattern = BLAS_GENERAL;
    return result;
}


static double blas_dot(const double *x, int n, int incx,
                       const double *y, int incy) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += x[i * incx] * y[i * incy];
    }
    return result;
}

static void blas_axpy(int n, double alpha, const double *x, int incx,
                      double *y, int incy) {
    for (int i = 0; i < n; i++) {
        y[i * incy] += alpha * x[i * incx];
    }
}



static void blas_ger(int m, int n, double alpha, 
                     const double *x, int incx,
                     const double *y, int incy,
                     double *A, int ldA) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * ldA + j] += alpha * x[i * incx] * y[j * incy];
        }
    }
}

static void blas_gemv(char trans, int m, int n, double alpha,
                      const double *A, int ldA,
                      const double *x, int incx, double beta,
                      double *y, int incy) {
    if (trans == 'N') {
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A[i * ldA + j] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    } else {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                sum += A[i * ldA + j] * x[i * incx];
            }
            y[j * incy] = alpha * sum + beta * y[j * incy];
        }
    }
}



static void blas_gemm(char transA, char transB, int m, int n, int k,
                      double alpha, const double *A, int ldA,
                      const double *B, int ldB, double beta,
                      double *C, int ldC) {
    if (transA == 'N' && transB == 'N') {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int kk = 0; kk < k; kk++) {
                    sum += A[i * ldA + kk] * B[kk * ldB + j];
                }
                C[i * ldC + j] = alpha * sum + beta * C[i * ldC + j];
            }
        }
    }
}



static Matrix* handle_dot(const Matrix *A, const Matrix *B) {
    if (A->ndim != 1 || B->ndim != 1 || A->shape[0] != B->shape[0]) {
        return NULL;
    }
    
    double dot_result = blas_dot(A->data, A->shape[0], 1, B->data, 1);
    
    int scalar_shape[1] = {1};
    Matrix *result = matrix_create_nd(1, scalar_shape, "");
    if (result) {
        result->data[0] = dot_result;
    }
    
    return result;
}

static Matrix* handle_axpy(const Matrix *A, const Matrix *B) {
    if (A->ndim != 1 || B->ndim != 1 || A->shape[0] != B->shape[0]) {
        return NULL;
    }
    
    int shape[1] = {A->shape[0]};
    Matrix *result = matrix_create_nd(1, shape, "i");
    if (!result) return NULL;
    
    memcpy(result->data, A->data, sizeof(double) * A->shape[0]);
    blas_axpy(A->shape[0], 1.0, B->data, 1, result->data, 1);
    
    return result;
}

static Matrix* handle_ger(const Matrix *A, const Matrix *B,
                          const char *in1, const char *in2, const char *out) {
    if (A->ndim != 1 || B->ndim != 1) {
        return NULL;
    }
    
    int m, n;
    const double *x, *y;
    
    if (in1[0] == out[0]) {
        m = A->shape[0];
        n = B->shape[0];
        x = A->data;
        y = B->data;
    } else {
        m = B->shape[0];
        n = A->shape[0];
        x = B->data;
        y = A->data;
    }
    
    int out_shape[2] = {m, n};
    Matrix *result = matrix_create_nd(2, out_shape, out);
    if (!result) return NULL;
    
    blas_ger(m, n, 1.0, x, 1, y, 1, result->data, n);
    
    return result;
}

static Matrix* handle_gemv(const Matrix *A, const Matrix *B,
                           const char *out) {
    const Matrix *mat;
    const double *vec;
    int m, n;
    
    if (A->ndim == 2 && B->ndim == 1) {
        mat = A;
        vec = B->data;
        m = A->shape[0];
        n = A->shape[1];
    } else if (A->ndim == 1 && B->ndim == 2) {
        mat = B;
        vec = A->data;
        m = B->shape[0];
        n = B->shape[1];
    } else {
        return NULL;
    }
    
    int result_shape[1] = {(A->ndim == 2) ? m : n};
    Matrix *result = matrix_create_nd(1, result_shape, out);
    if (!result) return NULL;
    
    char trans = (A->ndim == 1) ? 'T' : 'N';
    blas_gemv(trans, m, n, 1.0, mat->data, n, vec, 1, 0.0, result->data, 1);
    
    return result;
}

static Matrix* handle_gemm(const Matrix *A, const Matrix *B, const char *out) {
    if (A->ndim != 2 || B->ndim != 2) {
        return NULL;
    }
    
    int m = A->shape[0];
    int n = B->shape[1];
    int k = A->shape[1];
    
    if (B->shape[0] != k) {
        return NULL;
    }
    
    int out_shape[2] = {m, n};
    Matrix *result = matrix_create_nd(2, out_shape, out);
    if (!result) return NULL;
    
    blas_gemm('N', 'N', m, n, k, 1.0, A->data, k, B->data, n, 0.0, result->data, n);
    
    return result;
}


Matrix* einsum(const char *notation, const Matrix *A, const Matrix *B) {
    if (!notation || !A || !B) return NULL;
    
    char in1[32], in2[32], out[32];
    parse_einsum_notation(notation, in1, in2, out);
    
    BLASAnalysis analysis = analyze_blas_pattern(notation, A, B);
    
    #ifdef PRINT_BLAS_ANALYSIS
    fprintf(stderr, "BLAS pattern analysis for '%s': ", notation);
    switch (analysis.pattern) {
        case BLAS_DOT:    fprintf(stderr, "DOT\n"); break;
        case BLAS_AXPY:   fprintf(stderr, "AXPY\n"); break;
        case BLAS_GER:    fprintf(stderr, "GER\n"); break;
        case BLAS_GEMV:   fprintf(stderr, "GEMV\n"); break;
        case BLAS_GEMM:   fprintf(stderr, "GEMM\n"); break;
        case BLAS_GENERAL:fprintf(stderr, "GENERAL (NOT SUPPORTED)\n"); break;
    }
    #endif
    
    switch (analysis.pattern) {
        case BLAS_DOT:
            return handle_dot(A, B);
        case BLAS_AXPY:
            return handle_axpy(A, B);
        case BLAS_GER:
            return handle_ger(A, B, in1, in2, out);
        case BLAS_GEMV:
            return handle_gemv(A, B, out);
        case BLAS_GEMM:
            return handle_gemm(A, B, out);
        case BLAS_GENERAL:
        default:
            fprintf(stderr, "Error: Pattern not supported by BLAS implementation: %s\n", notation);
            return NULL;
    }
}
