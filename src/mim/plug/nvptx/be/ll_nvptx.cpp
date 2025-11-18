#include "mim/plug/nvptx/be/ll_nvptx.h"

#include <mim/plug/clos/clos.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>

using namespace std::string_literals;

namespace mim::ll {

namespace core = mim::plug::core;
namespace gpu  = mim::plug::gpu;
namespace mem  = mim::plug::mem;

namespace nvptx {

class Emitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    Emitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_emitter", ostream) {}

    virtual std::optional<std::string> isa_device_intrinsic(BB&, const Def*) override;
};

std::optional<std::string> Emitter::isa_device_intrinsic(BB& bb, const Def* def) {
    auto name = id(def);
    std::string op;

    if (auto malloc = Axm::isa<mem::malloc>(def)) {
        auto [Ta, msi]             = malloc->uncurry_args<2>();
        auto [pointee, addr_space] = Ta->projs<2>();
        auto address_space         = Lit::as<plug::mem::AddrSpace>(addr_space);

        switch (address_space) {
            case mem::AddrSpace::Generic: return std::nullopt;
            case mem::AddrSpace::Global: break;
            case mem::AddrSpace::Texture: error("malloc cannot be used in texture memory");
            case mem::AddrSpace::Shared: error("malloc cannot be used in shared memory");
            case mem::AddrSpace::Constant: error("malloc cannot be used in constant memory");
            case mem::AddrSpace::Local: error("malloc cannot be used in local memory");
            default: fe::unreachable();
        }

        declare("i32 @cuMemAlloc_v2(ptr, i64)");

        emit_unsafe(malloc->arg(0));
        auto alloc_ptr = bb.assign(name + "ptr", "alloca i64");

        auto size  = emit(malloc->arg(1));
        auto ptr_t = convert(Axm::as<mem::Ptr>(def->proj(1)->type()));

        auto alloc_res = bb.assign(name + "res", "call i32 @cuMemAlloc_v2(ptr {}, i64 {})", alloc_ptr, size);
        auto raw_ptr   = bb.assign(name + "i64raw", "load i64, i64* {}", alloc_ptr);
        auto ok        = bb.assign(name + "ok", "icmp eq i32 {}, 0", alloc_res);

        auto ptr_i64 = bb.assign(name + "i64", "select i1 {}, i64 {}, i64 0", ok, raw_ptr);
        return bb.assign(name, "inttoptr i64 {} to {}", ptr_i64, ptr_t);
    } else if (auto copy_mem_to_device = Axm::isa<gpu::copy_mem_to_device>(def)) {
        declare("i32 @cuMemcpyHtoD_v2(i64, ptr, i64)");

        auto [type]    = copy_mem_to_device->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_mem_to_device->arg(0));
        auto host_ptr = emit(copy_mem_to_device->arg(1));
        auto dev_ptr  = emit(copy_mem_to_device->arg(2));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));

        auto copy_res
            = bb.assign(name + "res", "call i32 @cuMemcpyHtoD_v2(ptr {}, ptr {}, i64 {})", dev_ptr, host_ptr, size);
        // TODO: error handling
        return copy_res;
    }

    return std::nullopt;
}

void emit(World& world, std::ostream& ostream) {
    Emitter emitter(world, ostream);
    emitter.run();
}

} // namespace nvptx

} // namespace mim::ll
