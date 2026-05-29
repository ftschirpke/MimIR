#include "mim/plug/nvptx/be/hcf_adapter.h"
#include "mim/plug/nvptx/be/hcf_container.hpp"

namespace mim::ll::pcuda {

namespace {

const char* param_type_name(HCFParamType t) {
    switch (t) {
        case HCFParamType::Integer:       return "integer";
        case HCFParamType::FloatingPoint: return "floating-point";
        case HCFParamType::Pointer:       return "pointer";
        case HCFParamType::OtherByValue:  return "other-by-value";
    }
    return "other-by-value";
}

} // namespace

std::string HCFBuilder::serialize() const {
    hcf::hcf_container hcf;
    auto* root = hcf.root_node();

    root->set("object-id", std::to_string(object_id_));
    root->set("generator", generator_);

    // images / llvm-ir.global
    auto* images   = root->add_subnode("images");
    auto* llvm_img = images->add_subnode("llvm-ir.global");
    llvm_img->set("variant", "global-module");
    llvm_img->set("format", "llvm-ir");
    llvm_img->set_as_list("exported-symbols", exported_);
    llvm_img->set_as_list("imported-symbols", imported_);
    hcf.attach_binary_content(llvm_img, device_bitcode_);

    // kernels
    auto* kernels_node = root->add_subnode("kernels");
    for (const auto& k : kernels_) {
        auto* kn = kernels_node->add_subnode(k.name);

        kn->set_as_list("image-providers", {std::string{"llvm-ir.global"}});

        auto* hsps = kn->add_subnode("host-side-parameter-sizes");
        for (std::size_t i = 0; i < k.host_side_parameter_sizes.size(); ++i)
            hsps->set(std::to_string(i), std::to_string(k.host_side_parameter_sizes[i]));

        // Empty but present — the runtime parser expects these subnodes.
        kn->add_subnode("compile-flags");
        kn->add_subnode("compile-options");

        auto* params = kn->add_subnode("parameters");
        for (std::size_t i = 0; i < k.parameters.size(); ++i) {
            const auto& p = k.parameters[i];
            auto* pn = params->add_subnode(std::to_string(i));
            pn->set("byte-offset", std::to_string(p.byte_offset));
            pn->set("byte-size", std::to_string(p.byte_size));
            pn->set("original-index", std::to_string(p.original_index));
            pn->set("type", param_type_name(p.type));
            auto* ann = pn->add_subnode("annotations");
            for (const auto& a : p.annotations) ann->set(a, "1");
        }
    }

    return hcf.serialize();
}

} // namespace mim::ll::pcuda
