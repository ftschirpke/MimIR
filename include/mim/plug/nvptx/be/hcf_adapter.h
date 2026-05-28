#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace mim::ll::pcuda {

/// HCF (Heterogeneous Container Format) adapter for kernel metadata generation
/// Provides utilities for creating HCF-compatible kernel metadata structures
/// that enable AdaptiveCpp's kernel cache to discover and JIT-compile kernels

/// Represents metadata for a single kernel
struct KernelMetadata {
    std::string name;                    // Kernel function name
    std::string target_backend;          // "ptx", "amdgpu", "spirv", "host", "metal"
    std::vector<std::string> arg_types;  // Argument types for parameter canonicalization
    int32_t warp_size;                   // SIMD lane count (32 for NVIDIA, 64 for AMD)
    int32_t max_threads_per_block;       // Device capability
    int32_t shared_mem_per_block;        // Device shared memory in bytes
};

/// HCF metadata container for kernel cache registration
class HCFMetadata {
public:
    /// Create a new HCF metadata container
    HCFMetadata();

    /// Add kernel metadata to the container
    void add_kernel(const KernelMetadata& metadata);

    /// Generate HCF readable header (text portion)
    /// Returns the textual HCF metadata format
    std::string generate_readable_header() const;

    /// Generate minimal HCF for kernel registration
    /// Creates a self-contained HCF string that can be embedded in host code
    /// and registered with __acpp_register_hcf()
    std::string generate_hcf_string() const;

    /// Generate HCF registration code
    /// Returns C code that registers the HCF at program startup
    std::string generate_registration_code() const;

private:
    std::vector<KernelMetadata> kernels_;
    std::map<std::string, size_t> kernel_ids_;
};

/// Generate kernel launch wrapper for pCUDA runtime
/// Creates a C function that wraps a kernel launch using pCUDA's
/// pcudaLaunchKernel or equivalent
std::string generate_kernel_launch_wrapper(
    const std::string& kernel_name,
    const std::vector<std::string>& arg_types,
    int32_t block_size_x,
    int32_t block_size_y,
    int32_t block_size_z
);

/// Convert LLVM IR function signature to HCF parameter format
/// Parses function signature and returns canonical argument type list
std::vector<std::string> extract_kernel_arguments(const std::string& llvm_function_sig);

} // namespace mim::ll::pcuda
