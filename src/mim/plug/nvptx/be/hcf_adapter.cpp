#include "mim/plug/nvptx/be/hcf_adapter.h"

#include <sstream>
#include <iomanip>
#include <algorithm>

namespace mim::ll::pcuda {

HCFMetadata::HCFMetadata() = default;

void HCFMetadata::add_kernel(const KernelMetadata& metadata) {
    kernel_ids_[metadata.name] = kernels_.size();
    kernels_.push_back(metadata);
}

std::string HCFMetadata::generate_readable_header() const {
    std::ostringstream oss;

    // HCF root properties
    oss << "hcf_version = 1\n";
    oss << "num_kernels = " << kernels_.size() << "\n";
    oss << "target_backends = pcuda\n";

    // Generate kernel metadata nodes
    for (size_t i = 0; i < kernels_.size(); ++i) {
        const auto& k = kernels_[i];
        oss << "{.kernel_" << i << "\n";
        oss << "  name = " << k.name << "\n";
        oss << "  backend = " << k.target_backend << "\n";
        oss << "  warp_size = " << k.warp_size << "\n";
        oss << "  max_threads_per_block = " << k.max_threads_per_block << "\n";
        oss << "  shared_memory_per_block = " << k.shared_mem_per_block << "\n";

        // Argument metadata
        if (!k.arg_types.empty()) {
            oss << "  num_args = " << k.arg_types.size() << "\n";
            for (size_t j = 0; j < k.arg_types.size(); ++j) {
                oss << "  arg_" << j << "_type = " << k.arg_types[j] << "\n";
            }
        } else {
            oss << "  num_args = 0\n";
        }
        oss << "}.kernel_" << i << "\n";
    }

    return oss.str();
}

std::string HCFMetadata::generate_hcf_string() const {
    std::ostringstream oss;

    // Generate readable header
    oss << generate_readable_header();

    // HCF binary separator (no binary data in this simple implementation)
    oss << "__acpp_hcf_binary_appendix";

    return oss.str();
}

std::string HCFMetadata::generate_registration_code() const {
    std::ostringstream oss;

    // Generate C code for HCF registration
    oss << "// Auto-generated HCF kernel registration code\n";
    oss << "#include <cstring>\n";
    oss << "#include <cstddef>\n\n";

    // Generate HCF content as a C string
    std::string hcf_content = generate_hcf_string();

    // Escape the HCF string for C source
    oss << "static const char __acpp_local_sscp_hcf_content[] = {\n";
    for (size_t i = 0; i < hcf_content.size(); ++i) {
        unsigned char c = hcf_content[i];
        oss << "  0x" << std::hex << std::setw(2) << std::setfill('0') << (int)c;
        if (i + 1 < hcf_content.size()) oss << ",";
        if ((i + 1) % 16 == 0) oss << "\n";
    }
    oss << std::dec << "\n};\n\n";

    // Size constant
    oss << "static const std::size_t __acpp_local_sscp_hcf_object_size = "
        << hcf_content.size() << ";\n\n";

    // Object ID (placeholder, set by runtime)
    oss << "static const std::size_t __acpp_local_sscp_hcf_object_id = 0;\n\n";

    // Registration helper if AdaptiveCpp headers are available
    oss << "// HCF registration will be handled by AdaptiveCpp runtime\n";
    oss << "// Ensure __acpp_register_hcf() is called with the above content\n";

    return oss.str();
}

std::string generate_kernel_launch_wrapper(
    const std::string& kernel_name,
    const std::vector<std::string>& arg_types,
    int32_t block_size_x,
    int32_t block_size_y,
    int32_t block_size_z
) {
    std::ostringstream oss;

    oss << "// pCUDA kernel launch wrapper for " << kernel_name << "\n";
    oss << "static pcudaError_t launch_" << kernel_name << "(\n";
    oss << "    int32_t device,\n";
    oss << "    int32_t grid_x, int32_t grid_y, int32_t grid_z,\n";
    oss << "    int32_t shared_mem,\n";
    oss << "    pcudaStream_t stream";

    for (size_t i = 0; i < arg_types.size(); ++i) {
        oss << ",\n    " << arg_types[i] << " arg" << i;
    }

    oss << "\n) {\n";
    oss << "    return pcudaLaunchKernel(\n";
    oss << "        kernel_name_to_ptr(\"" << kernel_name << "\"),\n";
    oss << "        grid_x, grid_y, grid_z,\n";
    oss << "        " << block_size_x << ", " << block_size_y << ", " << block_size_z << ",\n";
    oss << "        shared_mem,\n";
    oss << "        stream,\n";
    oss << "        nullptr  // args array would be populated here\n";
    oss << "    );\n";
    oss << "}\n\n";

    return oss.str();
}

std::vector<std::string> extract_kernel_arguments(const std::string& llvm_function_sig) {
    std::vector<std::string> args;

    // Simple parser for LLVM function signatures
    // Format: define spir_kernel return_type @kernel_name(arg1_type arg1, arg2_type arg2, ...)
    // This is a simplified implementation - a full parser would handle complex types

    size_t paren_start = llvm_function_sig.find('(');
    size_t paren_end = llvm_function_sig.rfind(')');

    if (paren_start == std::string::npos || paren_end == std::string::npos) {
        return args;
    }

    std::string params = llvm_function_sig.substr(paren_start + 1, paren_end - paren_start - 1);

    // Split by comma and extract type of each parameter
    size_t pos = 0;
    while (pos < params.size()) {
        // Skip whitespace
        while (pos < params.size() && std::isspace(params[pos])) ++pos;

        if (pos >= params.size()) break;

        // Find end of type (ends at space or comma)
        size_t type_end = pos;
        while (type_end < params.size() && !std::isspace(params[type_end]) && params[type_end] != ',') {
            ++type_end;
        }

        if (type_end > pos) {
            args.push_back(params.substr(pos, type_end - pos));
        }

        // Skip to next parameter
        while (pos < params.size() && params[pos] != ',') ++pos;
        if (pos < params.size()) ++pos; // skip comma
    }

    return args;
}

} // namespace mim::ll::pcuda
