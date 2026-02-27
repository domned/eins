#ifndef EINSUM_H
#define EINSUM_H

#include <stdint.h>

/* Type definition for index bitmap (26 bits for 'a'..'z') */
typedef uint32_t IndexBitmap;

/* Matrix structure for N-dimensional tensors */
typedef struct {
    double *data;       /* contiguous data buffer in row-major order */
    int ndim;           /* number of dimensions */
    int *shape;         /* array of dimension sizes, length ndim */
    IndexBitmap indices;/* bitmap of which indices (a-z) this matrix uses */
} Matrix;

/* ===== CORE FUNCTIONS ===== */

/* Convert a literal string (e.g., "ij") to a bitmap */
IndexBitmap literal_to_bitmap(const char *lit);

/* Parse einsum notation "in1,in2->out" into separate parts */
void parse_einsum_notation(const char *notation, char *in1, char *in2, char *out);

/* Create an N-dimensional matrix with given shape and index labels */
Matrix* matrix_create_nd(int ndim, const int *shape, const char *indices_str);

/* Free a matrix and its allocations */
void matrix_free(Matrix *m);

/* Access N-dimensional matrix element */
double matrix_get_nd(const Matrix *m, const int *idx);
void matrix_set_nd(Matrix *m, const int *idx, double v);

/* Access 2-D matrix element */
double matrix_get(const Matrix *m, int i, int j);
void matrix_set(Matrix *m, int i, int j, double v);

/* Advance multi-dimensional index in row-major order */
void next_indices(int *idx, const int *shape, int length, int *finished);

/* Print matrix contents */
void matrix_print(const Matrix *m);

/* ===== EINSUM OPERATION ===== */

/* Main einsum function: performs Einstein summation on two tensors */
Matrix* einsum(const char *notation, const Matrix *A, const Matrix *B);

#endif /* EINSUM_H */
