#include "mim/plug/nvptx/be/ll_nvptx.h"

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/nvptx/nvptx.h>

namespace mim::ll::nvptx {

namespace mem  = mim::plug::mem;
namespace gpu  = mim::plug::gpu;
namespace core = mim::plug::core;

class HostEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    HostEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_host_emitter", ostream) {}

    bool is_to_emit() override;
    void start() override;

    void emit_epilogue(Lam*) override;
    std::string prepare() override;

    std::optional<std::string> isa_targetspecific_intrinsic(BB&, const Def*) override;

protected:
    std::string convert(const Def*) override;

private:
    static constexpr std::string_view ctx_name           = "@mimir_cu_ctx";
    static constexpr std::string_view mod_name           = "@mimir_cu_mod";
    static constexpr std::string_view fatbin_name        = "@fatbin_fname";
    static constexpr std::string_view fatbin_value       = "mimir.fatbin";
    static constexpr std::string_view kernel_name_prefix = "@kname.";

    LamMap<int> kernel_ids;
};

class DeviceEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    DeviceEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_device_emitter", ostream) {}

    bool is_to_emit() override;

    std::string prepare() override;
};

static bool is_gpu_type(const Def* type) {
    if (auto m = Axm::isa<mem::M>(type)) {
        auto addr_space = m->arg();
        auto as         = Lit::as<nat_t>(addr_space);
        if (as == 2) error("NVPTX does not support working in address space 2");
        // TODO: this might not be error-proof e.g. when host writes to constant memory
        return as > 2;
    }
    if (auto sigma = type->isa<Sigma>()) {
        for (auto op : sigma->ops())
            if (is_gpu_type(op)) return true;
    }
    return false;
}

static bool is_gpu_code(Lam* lam) {
    return std::ranges::any_of(lam->vars(), [](auto var) { return is_gpu_type(var->type()); });
}

static bool is_kernel(Lam* lam) {
    if (!lam->is_external()) return false;
    if (!is_gpu_code(lam)) return false;
    // TODO: better kernel detection
    return true;
}

bool HostEmitter::is_to_emit() { return !is_gpu_code(root()); }

void HostEmitter::start() {
    DefSet done;
    for (auto mut : world().externals()) {
        if (auto lam = Lam::isa_mut_cn(mut.second)) {
            if (is_kernel(lam)) {
                assert(!kernel_ids.contains(lam));
                auto kid        = kernel_ids.size();
                kernel_ids[lam] = kid;
                auto name       = id(lam).substr(1);
                print(vars_decls_, "{}{} = private constant [{} x i8] c\"{}\\00\"\n", kernel_name_prefix, kid,
                      name.size() + 1, name);
            }
        }
    }
    Super::start();
}

static void emit_cu_error_handling(BB& bb, const std::string& cu_result) {
    // TODO: implement
    return;
}

std::string HostEmitter::convert(const Def* type) {
    if (auto ptr = Axm::isa<mem::Ptr>(type)) {
        auto [_, addr_space] = ptr->args<2>();
        auto lit             = Lit::isa(addr_space);
        if (lit.value_or(0L) != 0) {
            // NVIDIA treats all device pointers as i64s in host code
            return "i64";
        }
    }
    return Super::convert(type);
}

std::string HostEmitter::prepare() {
    auto name = Super::prepare();

    if (id(root()) != "@main") return name;

    auto vname = '%' + name;
    auto& bb   = lam2bb_[root()];

    auto dev_num   = 0; // TODO: consider parameterizing this
    auto ctx_flags = 0; // TODO: consider parameterizing this

    declare("i32 @cuInit(i32)");
    auto init_res = bb.assign(vname + "_init_res", "call i32 @cuInit(i32 0)");
    emit_cu_error_handling(bb, init_res);

    declare("i32 @cuDeviceGet(ptr, i32)");
    auto dev_ptr     = bb.assign(vname + "_dev_ptr", "alloca i32");
    auto dev_get_res = bb.assign(vname + "_get_res", "call i32 @cuDeviceGet(ptr {}, i32 {})", dev_ptr, dev_num);
    emit_cu_error_handling(bb, dev_get_res);

    declare("i32 @cuCtxCreate_v4(ptr, ptr, i32, i32)");
    print(vars_decls_, "{} = global ptr null\n", ctx_name);
    auto dev     = bb.assign(vname + "_dev", "load i32, ptr {}", dev_ptr);
    auto ctx_res = bb.assign(vname + "_ctx_res", "call i32 @cuCtxCreate_v4(ptr {}, ptr null, i32 {}, i32 {})", ctx_name,
                             ctx_flags, dev);
    emit_cu_error_handling(bb, ctx_res);

    // TODO: instead, load module using:  declare("i32 @cuModuleLoadFatBinary(ptr, ptr)");
    declare("i32 @cuModuleLoad(ptr, ptr)");
    print(vars_decls_, "{} = global ptr null\n", mod_name);
    print(vars_decls_, "{} = private constant [{} x i8] c\"{}\\00\"\n", fatbin_name, fatbin_value.size() + 1,
          fatbin_value);
    auto mod_res = bb.assign(vname + "_mod_res", "call i32 @cuModuleLoad(ptr {}, ptr {})", mod_name, fatbin_name);
    emit_cu_error_handling(bb, mod_res);

    return name;
}

void HostEmitter::emit_epilogue(Lam* lam) {
    Super::emit_epilogue(lam);
    if (id(lam) == "@main") {
        auto name = "%" + lam->unique_name();
        auto& bb  = lam2bb_[lam];

        declare("i32 @cuModuleUnload(ptr)");
        auto mod            = bb.assign(name + "_mod", "load ptr, ptr {}", mod_name);
        auto mod_unload_res = bb.assign(name + "_mod_unload_res", "call i32 @cuModuleUnload(ptr {})", mod);
        emit_cu_error_handling(bb, mod_unload_res);

        declare("i32 @cuCtxDestroy_v2(ptr)");
        auto ctx             = bb.assign(name + "_ctx", "load ptr, ptr {}", ctx_name);
        auto ctx_destroy_res = bb.assign(name + "_ctx_destroy_res", "call i32 @cuCtxDestroy_v2(ptr {})", ctx);
        emit_cu_error_handling(bb, ctx_destroy_res);
    }
}

std::optional<std::string> HostEmitter::isa_targetspecific_intrinsic(BB& bb, const Def* def) {
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
            default: fe::unreachable();
        }

        declare("i32 @cuMemAlloc_v2(ptr, i64)");

        emit_unsafe(malloc->arg(0));
        auto size  = emit(malloc->arg(1));
        auto ptr_t = convert(Axm::as<mem::Ptr>(def->proj(1)->type()));

        auto alloc_ptr = bb.assign(name + "ptr", "alloca {}", ptr_t);
        auto alloc_res = bb.assign(name + "res", "call i32 @cuMemAlloc_v2(ptr {}, i64 {})", alloc_ptr, size);
        emit_cu_error_handling(bb, alloc_res);
        return bb.assign(name, "load {}, {}* {}", ptr_t, ptr_t, alloc_ptr);
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
            default: fe::unreachable();
        }

        declare("i32 @cuMemFree_v2(i64)");

        emit_unsafe(free->arg(0));
        auto ptr = emit(free->arg(1));

        auto free_res = bb.assign(name + "res", "call i32 @cuMemFree_v2(i64 {})", ptr);
        emit_cu_error_handling(bb, free_res);
        return free_res;
    } else if (auto copy_mem_to_device = Axm::isa<gpu::copy_mem_to_device>(def)) {
        declare("i32 @cuMemcpyHtoD_v2(i64, ptr, i64)");

        auto [type]    = copy_mem_to_device->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_mem_to_device->arg(0));
        emit_unsafe(copy_mem_to_device->arg(1));
        auto host_ptr = emit(copy_mem_to_device->arg(2));
        auto dev_ptr  = emit(copy_mem_to_device->arg(3));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));

        auto copy_res
            = bb.assign(name + "res", "call i32 @cuMemcpyHtoD_v2(i64 {}, ptr {}, i64 {})", dev_ptr, host_ptr, size);
        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto copy_mem_to_host = Axm::isa<gpu::copy_mem_to_host>(def)) {
        declare("i32 @cuMemcpyDtoH_v2(ptr, i64, i64)");

        auto [type]    = copy_mem_to_host->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_mem_to_host->arg(0));
        emit_unsafe(copy_mem_to_host->arg(1));
        auto dev_ptr  = emit(copy_mem_to_host->arg(2));
        auto host_ptr = emit(copy_mem_to_host->arg(3));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));

        auto copy_res
            = bb.assign(name + "res", "call i32 @cuMemcpyDtoH_v2(ptr {}, i64 {}, i64 {})", host_ptr, dev_ptr, size);
        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto launch = Axm::isa<gpu::launch>(def)) {
        declare("i32 @cuLaunchKernel(ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr)");

        auto n_warps   = emit(launch->decurry()->decurry()->arg(0));
        auto n_threads = emit(launch->decurry()->decurry()->arg(1));

        emit_unsafe(launch->arg(0));
        auto func     = emit(launch->arg(1));
        auto arg      = emit(launch->arg(2));
        auto arg_type = convert(launch->arg(2)->type());

        declare("i32 @cuModuleGetFunction(ptr, ptr, ptr)");

        auto lam = Lam::isa_mut_cn(launch->arg(1));
        if (!lam) error("kernel is not a lamda {}", func);
        if (!kernel_ids.contains(lam)) error("unknown kernel {}", lam);
        auto kid = kernel_ids[lam];

        auto mod_inner = bb.assign("%mod_inner", "load ptr, ptr {}", mod_name);

        auto func_ptr = bb.assign(name + "_funcptr", "alloca ptr");
        auto func_res = bb.assign(name + "_getfuncres", "call i32 @cuModuleGetFunction(ptr {}, ptr {}, ptr {}{})",
                                  func_ptr, mod_inner, kernel_name_prefix, kid);
        emit_cu_error_handling(bb, func_res);

        auto arg_wrap = bb.assign(name + "_arg_wrap", "alloca {}", arg_type);
        print(bb.body().emplace_back(), "store {} {}, ptr {}", arg_type, arg, arg_wrap);

        auto args_ptr = bb.assign(name + "_args_ptr", "alloca [1 x ptr]");
        print(bb.body().emplace_back(), "store ptr {}, ptr {}", arg_wrap, args_ptr);
        auto shared_mem_bytes = 0;      // TODO: add shared memory support
        auto stream           = "null"; // TODO: add support for CUDA streams
        auto func_inner       = bb.assign(name + "_func_inner", "load ptr, ptr {}", func_ptr);
        auto args_inner
            = bb.assign(name + "_args_inner", "getelementptr inbounds [1 x ptr], ptr {}, i64 0, i64 0", args_ptr);
        auto launch_res = bb.assign(name,
                                    "call i32 @cuLaunchKernel(ptr {}, i32 {}, i32 1, i32 1, i32 {}, i32 1, i32 1,"
                                    "i32 {}, ptr {}, ptr {}, ptr null)",
                                    func_inner, n_warps, n_threads, shared_mem_bytes, stream, args_inner);
        emit_cu_error_handling(bb, launch_res);
        return launch_res;
    }

    return std::nullopt;
}

bool DeviceEmitter::is_to_emit() { return is_gpu_code(root()); }

std::string DeviceEmitter::prepare() {
    auto is_kern = is_kernel(root());

    auto attrs = is_kern ? std::string("ptx_kernel") : "";
    auto sep   = attrs.empty() ? "" : " ";
    print(func_impls_, "define{}{} {} {}(", sep, attrs, convert_ret_pi(root()->type()->ret_pi()), id(root()));

    auto vars                          = root()->vars();
    std::optional<size_t> block_index  = {};
    std::optional<size_t> thread_index = {};
    auto count                         = 0;
    for (auto sep = ""; auto var : vars.view().rsubspan(1)) {
        count++;
        if (Axm::isa<mem::M>(var->type())) continue;
        if (is_kern && Idx::isa(var->type())) {
            if (!block_index.has_value()) {
                block_index = count - 1;
                continue;
            }
            if (!thread_index.has_value()) {
                thread_index = count - 1;
                continue;
            }
        }
        auto name    = id(var);
        locals_[var] = name;
        print(func_impls_, "{}{} {}", sep, convert(var->type()), name);
        sep = ", ";
    }

    print(func_impls_, ") {{\n");

    if (is_kern) {
        auto& bb = lam2bb_[root()];
        {
            // block index
            if (!block_index.has_value()) error("couldn't find the block index for kernel '{}'", root());
            auto var       = root()->var(block_index.value());
            auto name      = id(var);
            auto type      = var->type();
            auto type_name = convert(type);
            auto idx_lit   = Idx::isa_lit(type);
            locals_[var]   = name;
            declare("i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
            if (!idx_lit.has_value() || type_name == "i0") {
                // TODO: handling of a non-existing value and "i0" case
            } else if (type_name == "i32") {
                auto i32 = bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
            } else if (idx_lit.value() < (1u << 31)) {
                auto i32 = bb.assign(name + "i32", "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
                bb.assign(name, "trunc i32 {} to {}", i32, convert(type));
            } else {
                error("Warp ID too large, must fit into I32");
            }
        }
        {
            // thread index
            if (!thread_index.has_value()) error("couldn't find the thread index for kernel '{}'", root());
            auto var       = root()->var(thread_index.value());
            auto name      = id(var);
            auto type      = var->type();
            auto type_name = convert(type);
            auto idx_lit   = Idx::isa_lit(type);
            locals_[var]   = name;
            declare("i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
            if (!idx_lit.has_value() || type_name == "i0") {
                // TODO: handling of a non-existing value and "i0" case
            } else if (type_name == "i32") {
                auto i32 = bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
            } else if (idx_lit.value() < (1u << 31)) {
                auto i32 = bb.assign(name + "i32", "call i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
                bb.assign(name, "trunc i32 {} to {}", i32, type_name);
            } else {
                error("Warp ID too large, must fit into I32");
            }
        }
    }

    return root()->unique_name();
}

void emit_host(World& world, std::ostream& ostream) {
    HostEmitter emitter(world, ostream);
    emitter.run();
}

void emit_device(World& world, std::ostream& ostream) {
    DeviceEmitter emitter(world, ostream);
    emitter.run();
}

} // namespace mim::ll::nvptx
