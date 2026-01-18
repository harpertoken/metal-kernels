# Quick Reference: Metal Compute Kernels

## Cheat Sheet for GPU Computing in Swift

### 1. Initialize GPU

```swift
let compute = MetalCompute()
```

### 2. Available Kernels

#### Basic Operations
```swift
// Array operations
compute.arrayAddition(a: [Float], b: [Float]) -> [Float]
compute.arrayMultiply(a: [Float], b: [Float]) -> [Float]
compute.scaleArray(_ a: [Float], scale: Float) -> [Float]
compute.absoluteValue(_ a: [Float]) -> [Float]
compute.relu(_ a: [Float]) -> [Float]
```

#### CUDA → Metal Translation
```swift
// Fused multiply-add
compute.fusedMultiplyAdd(a: [Float], b: [Float], c: [Float]) -> [Float]

// Sum reduction (CUDA __shared__ pattern)
compute.sumReduction(input: [Float]) -> [Float]
```

#### Image Processing
```swift
// 2D Convolution
compute.convolution2D(
    input: [Float],
    kernel: [Float],
    width: Int,
    height: Int
) -> [Float]
```

#### Machine Learning
```swift
// Activations
compute.sigmoid(_ input: [Float]) -> [Float]
compute.tanh(_ input: [Float]) -> [Float]
compute.gelu(_ input: [Float]) -> [Float]

// Convolution (mobile efficient)
compute.depthwiseConv2D(
    input: [Float],
    weights: [Float],
    bias: [Float],
    channels: Int,
    kernelSize: Int,
    inputSize: Int
) -> [Float]

// Matrix operations
compute.matrixMultiply(a: [Float], b: [Float], size: Int) -> [Float]
compute.batchedMatmul(
    a: [Float],
    b: [Float],
    batchSize: Int,
    m: Int,
    k: Int,
    n: Int
) -> [Float]
```

#### Advanced Patterns
```swift
// Exclusive scan (prefix sum)
compute.exclusiveScan(input: [Float]) -> [Float]

// Prefix sum starting at 0: [0, 1, 3, 6, 10]
// Input:                     [1, 2, 3, 4, 5]
```

### 3. Benchmarking

```swift
// Simple timing
compute.benchmark(name: "My Operation", iterations: 100) {
    _ = compute.arrayAddition(a: a, b: b)
}

// Detailed metrics (GPU utilization %)
let metrics = compute.benchmarkWithMetrics(
    name: "GPU Operation",
    iterations: 10
) {
    _ = compute.myKernel(input: data)
}

// Tune threadgroup sizes for your hardware
compute.benchmarkThreadgroupSizes(arraySize: 10000)
// Tests 32, 64, 128, 256 threads
```

### 4. CUDA → Metal Mapping

| CUDA | Metal | Metal Code |
|------|-------|-----------|
| `__global__` | `kernel` | `kernel void myFunc(...)` |
| `threadIdx.x` | `thread_index_in_threadgroup` | `uint lid [[thread_index_in_threadgroup]]` |
| `blockIdx.x` | N/A | Use `thread_position_in_grid` |
| Position | `thread_position_in_grid` | `uint id [[thread_position_in_grid]]` |
| `__shared__` | `threadgroup` | `threadgroup float* shared [[threadgroup(0)]]` |
| `__syncthreads()` | `threadgroup_barrier` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Buffer | `device` | `device float* buf [[buffer(0)]]` |
| Const | `constant` or `device const` | `device const float& val [[buffer(2)]]` |
| Result | `device` write | Write to output buffer |

### 5. Performance Expectations

**10,000 element array on M1/M2/M3:**
- GPU: 0.2-0.3 ms
- CPU: 2.5-3.5 ms
- **Speedup: 10-15×**

**64×64 matrix multiply:**
- GPU: 0.4-0.5 ms
- CPU: 35-40 ms
- **Speedup: 80-100×**

**Optimal threadgroup sizes:**
- M1/M2/M3: 64-128 threads
- M4: 128-256 threads
- iPhone 15 Pro: 32-64 threads

### 6. Common Patterns

#### Pattern: Element-wise operation
```swift
kernel void element_wise(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = activate(input[id]);  // sigmoid, relu, etc.
}
```

#### Pattern: Reduction
```swift
kernel void reduce(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    shared[lid] = input[id];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = 1; stride < 32; stride *= 2) {
        if (lid >= stride) {
            shared[lid] += shared[lid - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) output[id / 32] = shared[0];
}
```

#### Pattern: 2D operation
```swift
kernel void process_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& width [[buffer(2)]],
    device const uint& height [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]  // 2D coordinate
) {
    uint x = gid.x;
    uint y = gid.y;
    
    if (x >= width || y >= height) return;
    
    uint idx = y * width + x;
    output[idx] = process(input[idx]);
}
```

### 7. Debugging

```swift
// Add to kernel for debugging (prints to console)
// Note: Metal doesn't have printf; use assertions instead

// Check if input is valid
if (id >= max_size) return;  // Skip out-of-bounds threads

// Verify in Swift
let result = compute.arrayAddition(a: a, b: b)
assert(!result.contains { $0.isNaN }, "NaN in result!")
assert(!result.contains { $0.isInfinite }, "Infinity in result!")
```

### 8. Memory Layout

**Array storage:**
```swift
let arr: [Float] = [1, 2, 3, 4, 5]
// Memory: [1.0] [2.0] [3.0] [4.0] [5.0]
//           0     1     2     3     4    (indices)

// Buffer size in bytes
let bufferSize = arr.count * MemoryLayout<Float>.size  // 20 bytes for 5 floats
```

**2D array (row-major):**
```swift
let mat: [Float] = [
    1, 2, 3,   // Row 0
    4, 5, 6,   // Row 1
    7, 8, 9    // Row 2
]

// Access: mat[row * width + col]
//         mat[0 * 3 + 0] = 1
//         mat[1 * 3 + 2] = 6
//         mat[2 * 3 + 1] = 8
```

**Batch (4D tensor):**
```swift
let batch: [Float] = ... // [batch][height][width][channels]

// Index: batch_idx * (H*W*C) + h * (W*C) + w * C + c
let idx = b * (H*W*C) + y * (W*C) + x * C + channel
```

### 9. Optimization Tips

1. **Batch operations** - Reduce kernel launch overhead by 50-70%
2. **Coalesce memory** - Access sequential indices for cache hits
3. **Minimize divergence** - Threads should follow same code path
4. **Use shared memory** - Faster than global for threadgroup communication
5. **Profile early** - Use `benchmarkWithMetrics` to identify bottlenecks

### 10. Error Handling

```swift
// Check for GPU availability
guard let device = MTLCreateSystemDefaultDevice() else {
    print("Metal not available")
    return
}

// Wrap kernel calls for safety
do {
    let result = compute.myKernel(input: data)
    // Use result
} catch {
    print("GPU computation failed: \(error)")
}

// Validate results
let result = compute.arrayAddition(a: a, b: b)
let expected = zip(a, b).map { $0 + $1 }
assert(zip(result, expected).allSatisfy { abs($0 - $1) < 0.0001 },
       "GPU result doesn't match CPU!")
```

### 11. Deployment Checklist

- [ ] Tested on M1/M2/M3
- [ ] Benchmarked GPU vs CPU
- [ ] GPU utilization > 80%
- [ ] No NaN/infinity in output
- [ ] Memory properly freed
- [ ] Works in release build
- [ ] Profiled in Metal Debugger
- [ ] Works on target device (Mac/iPhone)

### 12. Resources

- **Metal Debugger**: Xcode → Window → Devices and Simulators
- **Metal Tutorial**: https://metal.shakir.io/
- **Apple Metal Docs**: https://developer.apple.com/metal/
- **CUDA Comparison**: https://docs.nvidia.com/cuda/

---

**Print this page for easy reference!**
