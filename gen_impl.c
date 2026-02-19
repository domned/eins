//--- Generating Code ---
// ========================================================
// Auto-generated implementation for einsum("ij,jc->ic")
// Shapes: A{7,3}, B{3,6}
// ========================================================

void einsum_generated(const double *A, const double *B, double *C) {
    for (int i = 0; i < 7; ++i) {
        for (int c = 0; c < 6; ++c) {
            for (int j = 0; j < 3; ++j) {
                    C[(i * 6) + c] += A[(i * 3) + j] * B[(j * 6) + c];
            }
        }
    }
}
