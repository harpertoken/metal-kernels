**Metal Compute Kernels for Swift**

Severity Package Platform Link
N/A metal-compute-kernels Apple Silicon (macOS / iOS) —

A Swift-based framework for GPU compute on Apple Silicon using Apple’s Metal API.

Metal Compute Kernels enables writing high-performance GPU kernels directly in Swift and provides direct CUDA-to-Metal execution model translations.

CUDA kernel example:

```cpp
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}
```

Metal translation:

```metal
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}
```

Key differences include the replacement of `blockIdx`/`threadIdx` with `thread_position_in_grid`, explicit `device` address space qualifiers with buffer bindings, and dispatch-controlled grid sizing instead of in-kernel bounds checks.