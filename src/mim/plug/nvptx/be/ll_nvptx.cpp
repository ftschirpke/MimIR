#include "mim/plug/nvptx/be/ll_nvptx.h"

#include <optional>

#include <mim/plug/clos/clos.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/nvptx/nvptx.h>

using namespace std::string_literals;

namespace mim::ll {

namespace core = mim::plug::core;
namespace gpu  = mim::plug::gpu;
namespace mem  = mim::plug::mem;

namespace nvptx {

class Emitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    enum class Target { Host, Device };

    Emitter(World& world, std::ostream& ostream, Target target)
        : Super(world, "llvm_nvptx_emitter", ostream)
        , target(target)
        , ctx_name(std::nullopt)
        , mod_name(std::nullopt) {}

    bool is_to_emit() override;

    std::string prepare() override;

    std::optional<std::string> isa_device_intrinsic(BB&, const Def*) override;

private:
    bool is_kernel();

    Target target;

    std::optional<std::string> ctx_name;
    std::optional<std::string> mod_name;
};

bool Emitter::is_kernel() {
    if (target != Target::Device) return false;
    if (root()->is_external()) {
        auto vars = root()->vars();
        if (vars.size() < 4) return false;
        if (!Axm::isa<gpu::M>(root()->var(0)->type())) return false;
        if (!Idx::isa(root()->var(1)->type())) return false;
        if (!Idx::isa(root()->var(2)->type())) return false;
        return true;
    }
    return {};
}

std::string Emitter::prepare() {
    auto is_kern = is_kernel();

    auto attrs = is_kern ? std::string("ptx_kernel") : "";
    auto sep   = attrs.empty() ? "" : " ";
    print(func_impls_, "define{}{} {} {}(", sep, attrs, convert_ret_pi(root()->type()->ret_pi()), id(root()));

    auto vars = root()->vars();
    auto i    = 0;
    for (auto sep = ""; auto var : vars.view().rsubspan(1)) {
        if (is_kern && i++ < 3) continue;
        if (Axm::isa<mem::M>(var->type())) continue;
        if (Axm::isa<gpu::M>(var->type())) continue;
        auto name    = id(var);
        locals_[var] = name;
        print(func_impls_, "{}{} {}", sep, convert(var->type()), name);
        sep = ", ";
    }

    print(func_impls_, ") {{\n");

    if (is_kern) {
        {
            // block index
            auto var     = root()->var(1);
            auto name    = id(var);
            locals_[var] = name;
            declare("i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
            // TODO: merge with existing methods
            print(func_impls_, "    {} = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n", name);
        }
        {
            // thread index
            auto var     = root()->var(2);
            auto name    = id(var);
            locals_[var] = name;
            declare("i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
            // TODO: merge with existing methods
            print(func_impls_, "    {} = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n", name);
        }
    }

    return root()->unique_name();
}

static bool is_gpu_type(const Def* type) {
    if (Axm::isa<gpu::M>(type)) return true;
    if (auto sigma = type->isa<Sigma>()) {
        for (auto op : sigma->ops())
            if (is_gpu_type(op)) return true;
    }
    return false;
}

bool Emitter::is_to_emit() {
    bool is_gpu_code = std::ranges::any_of(root()->vars(), [](auto var) { return is_gpu_type(var->type()); });
    switch (target) {
        case Target::Host: return !is_gpu_code;
        case Target::Device: return is_gpu_code;
    }
    fe::unreachable();
}

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
    } else if (auto free = Axm::isa<mem::free>(def)) {
        auto [Ta, msi]             = free->uncurry_args<2>();
        auto [pointee, addr_space] = Ta->projs<2>();
        auto address_space         = Lit::as<plug::mem::AddrSpace>(addr_space);

        switch (address_space) {
            case mem::AddrSpace::Generic: return std::nullopt;
            case mem::AddrSpace::Global: break;
            case mem::AddrSpace::Texture: error("free cannot be used in texture memory");
            case mem::AddrSpace::Shared: error("free cannot be used in shared memory");
            case mem::AddrSpace::Constant: error("free cannot be used in constant memory");
            case mem::AddrSpace::Local: error("free cannot be used in local memory");
            default: fe::unreachable();
        }

        declare("i32 @cuMemFree_v2(i64)");

        emit_unsafe(free->arg(0));
        auto ptr = emit(free->arg(1));

        auto free_res = bb.assign(name + "res", "call i32 @cuMemFree_v2(i64 {})", ptr);
        // TODO: error handling
        return free_res;
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
            = bb.assign(name + "res", "call i32 @cuMemcpyHtoD_v2(i64 {}, ptr {}, i64 {})", dev_ptr, host_ptr, size);
        // TODO: error handling
        return copy_res;
    } else if (auto copy_mem_to_host = Axm::isa<gpu::copy_mem_to_host>(def)) {
        declare("i32 @cuMemcpyDtoH_v2(ptr, i64, i64)");

        auto [type]    = copy_mem_to_host->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_mem_to_host->arg(0));
        auto dev_ptr  = emit(copy_mem_to_host->arg(1));
        auto host_ptr = emit(copy_mem_to_host->arg(2));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));

        auto copy_res
            = bb.assign(name + "res", "call i32 @cuMemcpyDtoH_v2(ptr {}, i64 {}, i64 {})", host_ptr, dev_ptr, size);
        // TODO: error handling
        return copy_res;
    } else if (auto launch = Axm::isa<gpu::launch>(def)) {
        if (!ctx_name.has_value() || !mod_name.has_value())
            error("Cannot launch a kernel without established CUDA context and module");

        declare("i32 @cuLaunchKernel(ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr)");

        emit_unsafe(launch->arg(0));
        auto n_warps   = emit(launch->arg(1));
        auto n_threads = emit(launch->arg(1));
        auto func      = emit(launch->arg(3));
        auto arg       = emit(launch->arg(4));
        auto arg_type  = convert(launch->arg(4)->type());

        declare("i32 @cuModuleGetFunction(ptr, ptr, ptr)");

        auto func_ptr = bb.assign(name + "_funcptr", "alloca ptr");
        auto func_res = bb.assign(name + "_getfuncres", "call i32 @cuModuleGetFunction(ptr {}, ptr {}, \"{}\")",
                                  func_ptr, mod_name.value(), func);
        // TODO: error handling
        auto args_ptr         = bb.assign(name + "_argsptr", "alloca ptr");
        auto args_store       = bb.assign(name + "_argsstore", "store ptr {}, ptr {}", arg, args_ptr);
        auto shared_mem_bytes = "0"; // TODO: add shared memory support
        auto stream           = "0"; // TODO: add support for CUDA streams
        auto launch_res       = bb.assign(name,
                                          "call i32 @cuLaunchKernel(ptr {}, i32 {}, i32 1, i32 1, i32 {}, i32 1, i32 1,"
                                                "i32 {}, i32 {}, ptr {}, ptr nullptr",
                                          func_ptr, n_warps, n_threads, shared_mem_bytes, stream, args_ptr);
        // TODO: error handling
        return launch_res;
    } else if (auto cu_init = Axm::isa<plug::nvptx::_cuInit>(def)) {
        declare("i32 @cuInit(i32)");

        emit_unsafe(cu_init->arg(0));
        return bb.assign(name, "call i32 @cuInit(i32 0)");
    } else if (auto cu_device_get_count = Axm::isa<plug::nvptx::_cuDeviceGetCount>(def)) {
        declare("i32 @cuDeviceGetCount(ptr)");

        emit_unsafe(cu_device_get_count->arg(0));
        auto dev_count_ptr = emit(cu_device_get_count->arg(1));
        // TODO: error handling
        return bb.assign(name, "call i32 @cuDeviceGetCount(ptr {})", dev_count_ptr);
    } else if (auto cu_device_get = Axm::isa<plug::nvptx::_cuDeviceGet>(def)) {
        declare("i32 @cuDeviceGet(ptr, i32)");

        emit_unsafe(cu_device_get->arg(0));
        auto dev_ptr = emit(cu_device_get->arg(1));
        auto dev_num = emit(cu_device_get->arg(2));
        // TODO: error handling
        return bb.assign(name, "call i32 @cuDeviceGet(ptr {}, i32 {})", dev_ptr, dev_num);
    } else if (auto cu_ctx_create = Axm::isa<plug::nvptx::_cuCtxCreate>(def)) {
        declare("i32 @cuCtxCreate_v4(ptr, ptr, i32, i32)");

        emit_unsafe(cu_ctx_create->arg(0));
        auto ctx_ptr = emit(cu_ctx_create->arg(1));
        auto flags   = emit(cu_ctx_create->arg(2));
        auto dev     = emit(cu_ctx_create->arg(3));

        ctx_name = ctx_ptr;
        // TODO: error handling
        return bb.assign(name, "call i32 @cuCtxCreate(ptr {}, ptr nullptr, i32 {}, i32 {})", ctx_ptr, flags, dev);
    } else if (auto cu_module_load_fat_binary = Axm::isa<plug::nvptx::_cuModuleLoadFatBinary>(def)) {
        declare("i32 @cuModuleLoadFatBinary(ptr, ptr)");

        emit_unsafe(cu_module_load_fat_binary->arg(0));
        auto mod_ptr = emit(cu_module_load_fat_binary->arg(1));
        auto data    = emit(cu_module_load_fat_binary->arg(2));

        mod_name = mod_ptr;
        // TODO: error handling
        return bb.assign(name, "call i32 @cuModuleLoadFatBinary(ptr {}, ptr {})", mod_ptr, data);
    } else if (auto cu_module_unload = Axm::isa<plug::nvptx::_cuModuleUnload>(def)) {
        declare("i32 @cuModuleUnload(ptr noundef)");

        emit_unsafe(cu_module_unload->arg(0));
        auto mod = emit(cu_module_unload->arg(1));

        mod_name = std::nullopt;
        // TODO: error handling
        return bb.assign(name, "call i32 @cuModuleUnload(ptr {})", mod);
    } else if (auto cu_ctx_destroy = Axm::isa<plug::nvptx::_cuCtxDestroy>(def)) {
        declare("i32 @cuCtxDestroy_v2(ptr noundef)");

        emit_unsafe(cu_ctx_destroy->arg(0));
        auto ctx = emit(cu_ctx_destroy->arg(1));

        ctx_name = std::nullopt;
        // TODO: error handling
        return bb.assign(name, "call i32 @cuCtxDestroy_v2(ptr {})", ctx);
    }
    return std::nullopt;
}

void emit_host(World& world, std::ostream& ostream) {
    Emitter emitter(world, ostream, Emitter::Target::Host);
    emitter.run();
}

void emit_device(World& world, std::ostream& ostream) {
    Emitter emitter(world, ostream, Emitter::Target::Device);
    emitter.run();
}

} // namespace nvptx

} // namespace mim::ll
