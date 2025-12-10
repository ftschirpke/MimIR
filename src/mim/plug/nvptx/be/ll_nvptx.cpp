#include "mim/plug/nvptx/be/ll_nvptx.h"

#include <optional>

#include <mim/plug/clos/clos.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/nvptx/nvptx.h>

#include "mim/util/print.h"

#include "absl/container/flat_hash_map.h"

using namespace std::string_literals;

namespace mim::ll {

namespace core = mim::plug::core;
namespace gpu  = mim::plug::gpu;
namespace mem  = mim::plug::mem;

namespace nvptx {

class HostEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    HostEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_host_emitter", ostream) {}

    bool is_to_emit() override;

    void start() override;
    std::string prepare() override;
    void emit_epilogue(Lam*) override;

    std::optional<std::string> isa_device_intrinsic(BB&, const Def*) override;

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

    std::optional<std::string> isa_device_intrinsic(BB&, const Def*) override;
};

// TODO: rethink kernel detection
// detect kernel by checking for the signature:  extern Cn[%gpu.M, Idx N, Idx M, T, Cn %gpu.M]
static bool is_kernel(Lam* lam) {
    if (!lam->is_external()) return false;

    auto vars = lam->vars();
    if (vars.size() != 5) return false;
    if (!Axm::isa<gpu::M>(lam->var(0)->type())) return false;
    if (!Idx::isa(lam->var(1)->type())) return false;
    if (!Idx::isa(lam->var(2)->type())) return false;
    return true;
}

static bool is_gpu_type(const Def* type) {
    if (Axm::isa<gpu::M>(type)) return true;
    if (auto sigma = type->isa<Sigma>()) {
        for (auto op : sigma->ops())
            if (is_gpu_type(op)) return true;
    }
    return false;
}

bool HostEmitter::is_to_emit() {
    bool is_gpu_code = std::ranges::any_of(root()->vars(), [](auto var) { return is_gpu_type(var->type()); });
    return !is_gpu_code;
}

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

std::optional<std::string> HostEmitter::isa_device_intrinsic(BB& bb, const Def* def) {
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
            case mem::AddrSpace::Local: error("free cannot be used in local memory");
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
        auto host_ptr = emit(copy_mem_to_device->arg(1));
        auto dev_ptr  = emit(copy_mem_to_device->arg(2));
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
        auto dev_ptr  = emit(copy_mem_to_host->arg(1));
        auto host_ptr = emit(copy_mem_to_host->arg(2));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));

        auto copy_res
            = bb.assign(name + "res", "call i32 @cuMemcpyDtoH_v2(ptr {}, i64 {}, i64 {})", host_ptr, dev_ptr, size);
        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto launch = Axm::isa<gpu::launch>(def)) {
        declare("i32 @cuLaunchKernel(ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr)");

        emit_unsafe(launch->arg(0));
        auto n_warps   = emit(launch->arg(1));
        auto n_threads = emit(launch->arg(2));
        auto func      = emit(launch->arg(3));
        auto arg       = emit(launch->arg(4));
        auto arg_type  = convert(launch->arg(4)->type());

        declare("i32 @cuModuleGetFunction(ptr, ptr, ptr)");

        auto lam = Lam::isa_mut_cn(launch->arg(3));
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

bool DeviceEmitter::is_to_emit() {
    bool is_gpu_code = std::ranges::any_of(root()->vars(), [](auto var) { return is_gpu_type(var->type()); });
    return is_gpu_code;
}

std::string DeviceEmitter::prepare() {
    auto is_kern = is_kernel(root());

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
        auto& bb = lam2bb_[root()];
        {
            // block index
            auto var  = root()->var(1);
            auto name = id(var);
            if (!name.starts_with("%_")) { // HACK: this is a bad way to check whether the argument is used later
                auto type      = var->type();
                auto type_name = convert(type);
                auto idx_lit   = Idx::isa_lit(type);
                locals_[var]   = name;
                declare("i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
                // HACK: the handling of a non-existing value and the "i0" case should be reconsidered
                if (!idx_lit.has_value() || type_name == "i32" || type_name == "i0") {
                    auto i32 = bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
                } else if (idx_lit.value() < (1u << 31)) {
                    auto i32 = bb.assign(name + "i32", "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
                    bb.assign(name, "trunc i32 {} to {}", i32, convert(type));
                } else {
                    error("Warp ID too large, must fit into I32");
                }
            }
        }
        {
            // thread index
            auto var  = root()->var(2);
            auto name = id(var);
            if (!name.starts_with("%_")) { // HACK: this is a bad way to check whether the argument is used later
                auto type      = var->type();
                auto type_name = convert(type);
                auto idx_lit   = Idx::isa_lit(type);
                locals_[var]   = name;
                declare("i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
                // HACK: the handling of a non-existing value and the "i0" case should be reconsidered
                if (!idx_lit.has_value() || type_name == "i32" || type_name == "i0") {
                    auto i32 = bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
                } else if (idx_lit.value() < (1u << 31)) {
                    auto i32 = bb.assign(name + "i32", "call i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
                    bb.assign(name, "trunc i32 {} to {}", i32, type_name);
                } else {
                    error("Warp ID too large, must fit into I32");
                }
            }
        }
        // {
        //     // TODO: remove, just for demonstration purposes
        //     declare("i32 @vprintf(ptr, ptr)");
        //     auto printf_arg      = id(root()->var(2));
        //     auto printf_arg_type = "i32";
        //     print(type_decls_, "%printf_args = type {{ {} }}\n", printf_arg_type);
        //     print(vars_decls_, "@welcome_message = private constant [15 x i8] c\"hi from t %2d\\0A\\00\"\n");
        //     auto printf_arg_buf       = bb.assign("%printf_buf", "alloca %printf_args");
        //     auto printf_arg_buf_inner = bb.assign(
        //         "%printf_args_inner", "getelementptr inbounds %printf_args, ptr {}, i32 0, i32 0", printf_arg_buf);
        //     print(bb.body().emplace_back(), "store {} {}, ptr {}", printf_arg_type, printf_arg,
        //     printf_arg_buf_inner); print(bb.body().emplace_back(), "call i32 @vprintf(ptr @welcome_message, ptr {})",
        //     printf_arg_buf);
        // }
    }

    return root()->unique_name();
}

std::optional<std::string> DeviceEmitter::isa_device_intrinsic(BB& bb, const Def* def) {
    auto name           = id(def);
    auto emit_gep_index = [&](const Def* index) {
        auto v_i = emit(index);
        auto t_i = convert(index->type());

        if (auto size = Idx::isa(index->type())) {
            if (auto w = Idx::size2bitwidth(size); w && *w < 64) {
                v_i = bb.assign(name + ".zext",
                                "zext {} {} to i{} ; add one more bit for gep index as it is treated as signed value",
                                t_i, v_i, *w + 1);
                t_i = "i" + std::to_string(*w + 1);
            }
        }

        return std::pair(v_i, t_i);
    };

    if (auto store = Axm::isa<gpu::store>(def)) {
        emit_unsafe(store->arg(0));
        auto v_ptr = emit(store->arg(1));
        auto v_val = emit(store->arg(2));
        auto t_ptr = convert(store->arg(1)->type());
        auto t_val = convert(store->arg(2)->type());
        print(bb.body().emplace_back(), "store {} {}, {} {}", t_val, v_val, t_ptr, v_ptr);
        return "";
    } else if (auto lea = Axm::isa<gpu::lea>(def)) {
        auto [ptr, i]  = lea->args<2>();
        auto pointee   = Axm::as<mem::Ptr>(ptr->type())->arg(0);
        auto v_ptr     = emit(ptr);
        auto t_pointee = convert(pointee);
        auto t_ptr     = convert(ptr->type());
        if (pointee->isa<Sigma>())
            return bb.assign(name, "getelementptr inbounds {}, {} {}, i64 0, i32 {}", t_pointee, t_ptr, v_ptr,
                             Lit::as(i));

        assert(pointee->isa<Arr>());
        auto [v_i, t_i] = emit_gep_index(i);

        return bb.assign(name, "getelementptr inbounds {}, {} {}, i64 0, {} {}", t_pointee, t_ptr, v_ptr, t_i, v_i);
    }
    return std::nullopt;
}

void emit_host(World& world, std::ostream& ostream) {
    HostEmitter emitter(world, ostream);
    emitter.run();
}

void emit_device(World& world, std::ostream& ostream) {
    DeviceEmitter emitter(world, ostream);
    emitter.run();
}

} // namespace nvptx

} // namespace mim::ll
