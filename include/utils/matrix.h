#ifndef MATRIX_H
#define MATRIX_H

void fill_random(float* matrix, int size);

void fill_zero(float* matrix, int size);

bool compare_matrices(const float* A, const float* B, int size, float tolerance = 1e-4f);

#endif