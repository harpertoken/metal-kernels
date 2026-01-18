# Contributing to Metal Compute Kernels

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/MetalKernels.git
   cd MetalKernels
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-kernel-name
   ```

3. **Build and test**
   ```bash
   swift build
   swift run MetalKernels
   ```

## Areas for Contribution

### 1. New Kernels

Add new Metal compute kernels for:
- RNN layers (LSTM, GRU)
- Transformer attention
- Quantization operations
- Custom loss functions
- Physics simulations

**Steps:**
1. Add Metal kernel code to the metalCode string in `MetalCompute.init()`
2. Add Swift host function in `MetalCompute` class
3. Add test case in main.swift
4. Document in README.md

### 2. Performance Optimization

- Profile existing kernels with Metal Debugger
- Optimize memory access patterns
- Reduce kernel launch overhead
- Add threadgroup memory optimizations
- Vectorize operations

### 3. Documentation

- Write kernel implementation guides
- Create usage examples
- Benchmark different chip generations
- Document best practices

### 4. Testing

- Add comprehensive test suite
- Benchmark against CPU and other GPU libraries
- Test on multiple Mac/iPad models
- Verify edge cases (NaN, infinity, zero)

### 5. Cross-platform Support

- Vulkan compute backend for non-Apple devices
- Metal on iOS, tvOS, macOS variants
- Support for older Metal versions

## Coding Style

### Swift Code

```swift
// Use clear variable names
let inputBuffer = device.makeBuffer(bytes: input, length: bufferSize)

// Add comments for complex logic
// Parallel reduction: log(n) steps with threadgroup_barrier
for stride in 1..<32 {
    if lid >= stride {
        shared[lid] += shared[lid - stride]
    }
    threadgroup_barrier(mem_flags::mem_threadgroup)
}

// Group related functions with MARK comments
// MARK: - Neural Network Layers
```

### Metal (MSL) Code

```metal
// Use clear parameter names and attributes
kernel void my_operation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& param [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    // Implementation
}

// Add comments for non-obvious operations
float normalized = (input[id] - mean) / sqrt(var + epsilon);
```

## Commit Message Format

```
[Category] Brief description (50 chars max)

Longer description explaining the change, why it was needed,
and any important implementation details. Wrap at 72 characters.

Related issues: #123
```

**Categories:**
- `[Feature]` - New kernel or capability
- `[Fix]` - Bug fix
- `[Perf]` - Performance improvement
- `[Docs]` - Documentation update
- `[Test]` - Test additions
- `[Refactor]` - Code reorganization

Example:
```
[Feature] Add LSTM kernel for sequence processing

Implements long short-term memory cell with peephole connections.
Uses 4 matrix multiplications per timestep. Supports batching.
Tested on M1/M2/M3 with 100x speedup over CPU.

Related issues: #42
```

## Pull Request Process

1. **Create PR with clear title and description**
   ```
   Title: Add LSTM kernel for RNN support
   
   Description:
   - Implements LSTM with peephole connections
   - Supports batch processing (up to 256)
   - ~50x faster than NumPy on M1
   - Tested on M1/M2/M3 and iPhone 15 Pro
   - Includes comprehensive benchmarks
   ```

2. **Ensure code builds**
   ```bash
   swift build -v
   swift run MetalKernels
   ```

3. **Add tests**
   - Add test case to main.swift
   - Include correctness check against CPU
   - Benchmark GPU vs CPU

4. **Update documentation**
   - Add kernel to appropriate section in README.md
   - Update performance table
   - Include usage example

5. **Wait for review**
   - Address feedback promptly
   - Make requested changes in new commits
   - Don't force-push after review starts

## Performance Benchmarking

When adding kernels, include:

```swift
print("📊 BENCHMARK: My Kernel")
compute.benchmarkWithMetrics(name: "My Kernel (GPU)", iterations: 10) {
    _ = compute.myKernel(input: testData)
}

let cpuStart = Date()
for _ in 0..<10 {
    _ = cpu_my_kernel(testData)
}
let cpuTime = Date().timeIntervalSince(cpuStart) / 10.0 * 1000
print("⏱️  My Kernel (CPU): \(String(format: "%.3f", cpuTime)) ms/iter")
```

Expected results section:
```
Speedup on M1: ~50x
Speedup on M2: ~50x
Speedup on M3: ~60x
Speedup on iPhone 15 Pro: ~40x
GPU Utilization: ~85%
```

## Debugging Tips

### Metal Shader Compilation Errors

- Check syntax (Metal is strict about C++ rules)
- Verify buffer indices are sequential
- Ensure threadgroup memory is allocated
- Look for reserved keyword conflicts

### Runtime Issues

- Check `MTLBuffer` sizes (multiply count × element size)
- Verify buffer indices match kernel signature
- Use `waitUntilCompleted()` for debugging
- Print thread counts before dispatch

### Performance Profiling

```bash
# Use Metal Debugger
# 1. Run app in Xcode
# 2. Xcode > Window > Devices and Simulators > Open Console
# 3. Click Metal Debugger in Debug Navigator
# 4. Capture frame and analyze
```

## Testing on Multiple Devices

Ideal test matrix:

- [ ] M1 MacBook Air
- [ ] M2 MacBook Pro
- [ ] M3 MacBook Pro
- [ ] M1/M2 iPad Pro
- [ ] iPhone 15 Pro (A17)

Or use:
- Apple Silicon Cloud Computing (AWS Graviton for simulation)
- BrowserStack for iOS testing

## Documentation Requirements

Every kernel needs:

1. **In-code comments**
   ```swift
   /// Matrix multiplication using tiled approach
   /// - Parameters:
   ///   - a: Left matrix (M×K)
   ///   - b: Right matrix (K×N)
   ///   - size: Dimension (assumes square matrices)
   /// - Returns: Result matrix (M×N)
   func tiledMatrixMultiply(a: [Float], b: [Float], size: Int) -> [Float]
   ```

2. **README.md entry** with category and example

3. **Performance table** with GPU/CPU times

4. **Use case description** explaining when to use

## Licensing

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: your.email@example.com

---

Thank you for contributing! 🙏
