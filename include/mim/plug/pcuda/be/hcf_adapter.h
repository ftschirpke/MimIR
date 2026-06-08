#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mim::ll::pcuda {

/// HCF (Heterogeneous Container Format) builder for the pCUDA backend.
///
/// Produces the exact text-plus-binary-appendix format that AdaptiveCpp's
/// runtime parses via `hipsycl::common::hcf_container::parse`, mirroring the
/// SSCP plugin's `generateHCF` (TargetSeparationPass.cpp:434-516). The output
/// is embedded into the host LLVM IR as `@__acpp_local_sscp_hcf_content`.
///
/// Layout:
///   root: object-id, generator
///     images / llvm-ir.global: variant=global-module, format=llvm-ir,
///                              exported-symbols (list), imported-symbols (list),
///                              __binary { start, size }
///     kernels / <name>: image-providers, host-side-parameter-sizes,
///                       compile-flags, compile-options,
///                       parameters / <i>: byte-offset, byte-size,
///                                         original-index, type, annotations
///   __acpp_hcf_binary_appendix<raw device bitcode bytes>

enum class HCFParamType { Integer, FloatingPoint, Pointer, OtherByValue };

struct HCFParam {
    std::size_t byte_offset;
    std::size_t byte_size;
    std::size_t original_index;
    HCFParamType type;
    std::vector<std::string> annotations;
};

struct HCFKernel {
    std::string name;
    std::vector<std::size_t> host_side_parameter_sizes;
    std::vector<HCFParam> parameters;
};

class HCFBuilder {
public:
    HCFBuilder() = default;

    void set_object_id(std::uint64_t id) { object_id_ = id; }
    void set_generator(std::string s) { generator_ = std::move(s); }

    /// Raw LLVM bitcode bytes (output of llvm-as on the device .ll).
    void set_device_bitcode(std::string bytes) { device_bitcode_ = std::move(bytes); }

    void set_exported_symbols(std::vector<std::string> syms) { exported_ = std::move(syms); }
    void set_imported_symbols(std::vector<std::string> syms) { imported_ = std::move(syms); }

    void add_kernel(HCFKernel k) { kernels_.push_back(std::move(k)); }

    std::uint64_t object_id() const { return object_id_; }
    const std::vector<HCFKernel>& kernels() const { return kernels_; }

    /// Serialize to the wire format that the AdaptiveCpp runtime accepts.
    std::string serialize() const;

private:
    std::uint64_t object_id_ = 0;
    std::string generator_ = "MimIR pCUDA backend";
    std::string device_bitcode_;
    std::vector<std::string> exported_;
    std::vector<std::string> imported_;
    std::vector<HCFKernel> kernels_;
};

} // namespace mim::ll::pcuda
