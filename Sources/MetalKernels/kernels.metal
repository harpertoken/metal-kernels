#include <metal_stdlib>
using namespace metal;

// Simple array addition kernel
kernel void add_arrays(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = a[id] + b[id];
}

// Vector multiply accumulate (GEMV-like)
kernel void matrix_vector_multiply(
    device const float* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    for (uint col = 0; col < cols; col++) {
        sum += matrix[row * cols + col] * vector[col];
    }
    result[row] = sum;
}
