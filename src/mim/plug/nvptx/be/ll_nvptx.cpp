#include "mim/plug/nvptx/be/ll_nvptx.h"

#include <optional>

#include <mim/plug/clos/clos.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/nvptx/nvptx.h>

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
        : Super(world, "llvm_nvptx_host_emitter", ostream)
        , ctx_name(std::nullopt)
        , mod_name(std::nullopt) {}

    bool is_to_emit() override;

    void start() override;
    std::string prepare() override;
    void emit_epilogue(Lam*) override;

    std::optional<std::string> isa_device_intrinsic(BB&, const Def*) override;

private:
    std::optional<std::string> ctx_name;
    std::optional<std::string> mod_name;
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
                ILOG("FRIEDRICH detected kernel '{}'", lam);
                assert(!kernel_ids.contains(lam));
                auto id         = kernel_ids.size();
                kernel_ids[lam] = id;
                auto name       = lam->unique_name();
                print(vars_decls_, "@.kname.{} = private constant [{} x i8] c\"{}\\00\"\n", id, name.size() + 1, name);
            } else if (id(lam) == "@main")
                ILOG("FRIEDRICH found main '{}'", lam);
            else
                ILOG("FRIEDRICH uncategorized '{}'", lam);
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

    declare("i32 @cuDeviceGetCount(ptr)");
    auto dev_count_ptr = bb.assign(vname + "_dev_count_ptr", "alloca i32");
    print(bb.body().emplace_back(), "store i32 0, ptr {}", dev_count_ptr);
    auto count_res = bb.assign(vname + "_count_res", "call i32 @cuDeviceGetCount(ptr {})", dev_count_ptr);
    emit_cu_error_handling(bb, count_res);

    declare("i32 @cuDeviceGet(ptr, i32)");
    auto dev_ptr = bb.assign(vname + "_dev_ptr", "alloca i32");
    print(bb.body().emplace_back(), "store i32 0, ptr {}", dev_count_ptr);
    auto dev_get_res = bb.assign(vname + "_get_res", "call i32 @cuDeviceGet(ptr {}, i32 {})", dev_ptr, dev_num);
    emit_cu_error_handling(bb, dev_get_res);

    declare("i32 @cuCtxCreate_v4(ptr, ptr, i32, i32)");
    auto ctx_ptr = bb.assign(vname + "_ctx_ptr", "alloca ptr");
    print(bb.body().emplace_back(), "store ptr null, ptr {}", ctx_ptr);
    auto dev     = bb.assign(vname + "_dev", "load i32, ptr {}", dev_ptr);
    auto ctx_res = bb.assign(vname + "_ctx_res", "call i32 @cuCtxCreate_v4(ptr {}, ptr null, i32 {}, i32 {})", ctx_ptr,
                             ctx_flags, dev);
    emit_cu_error_handling(bb, ctx_res);
    ctx_name = ctx_ptr;

    // TODO: instead, load module using:  declare("i32 @cuModuleLoadFatBinary(ptr, ptr)");
    declare("i32 @cuModuleLoad(ptr, ptr)");
    auto mod_ptr = bb.assign(vname + "_mod_ptr", "alloca ptr");
    print(bb.body().emplace_back(), "store ptr null, ptr {}", mod_ptr);
    print(vars_decls_, "@fatbin_fname = private constant [13 x i8] c\"mimir.fatbin\\00\"\n");
    auto mod_res = bb.assign(vname + "_mod_res", "call i32 @cuModuleLoad(ptr {}, ptr {})", mod_ptr, "@fatbin_fname");
    emit_cu_error_handling(bb, mod_res);
    mod_name = mod_ptr;

    return name;
}

void HostEmitter::emit_epilogue(Lam* lam) {
    Super::emit_epilogue(lam);
    if (id(lam) == "@main") {
        auto name = "%" + lam->unique_name();
        auto& bb  = lam2bb_[lam];

        if (mod_name.has_value()) {
            declare("i32 @cuModuleUnload(ptr)");
            auto mod            = bb.assign(name + "_mod", "load ptr, ptr {}", mod_name.value());
            auto mod_unload_res = bb.assign(name + "_mod_unload_res", "call i32 @cuModuleUnload(ptr {})", mod);
            emit_cu_error_handling(bb, mod_unload_res);
            mod_name = std::nullopt;
        }
        if (ctx_name.has_value()) {
            declare("i32 @cuCtxDestroy_v2(ptr)");
            auto ctx             = bb.assign(name + "_ctx", "load ptr, ptr {}", ctx_name.value());
            auto ctx_destroy_res = bb.assign(name + "_ctx_destroy_res", "call i32 @cuCtxDestroy_v2(ptr {})", ctx);
            emit_cu_error_handling(bb, ctx_destroy_res);
            ctx_name = std::nullopt;
        }
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
        auto alloc_ptr = bb.assign(name + "ptr", "alloca i64");

        auto size  = emit(malloc->arg(1));
        auto ptr_t = convert(Axm::as<mem::Ptr>(def->proj(1)->type()));

        auto alloc_res = bb.assign(name + "res", "call i32 @cuMemAlloc_v2(ptr {}, i64 {})", alloc_ptr, size);
        // TODO: reconsider using:  emit_cu_error_handling(bb, alloc_res);
        auto raw_ptr = bb.assign(name + "i64raw", "load i64, i64* {}", alloc_ptr);
        auto ok      = bb.assign(name + "ok", "icmp eq i32 {}, 0", alloc_res);
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
        emit_cu_error_handling(bb, free_res);
        return free_res;
    } else if (auto copy_mem_to_device = Axm::isa<gpu::copy_mem_to_device>(def)) {
        declare("i32 @cuMemcpyHtoD_v2(i64, ptr, i64)");

        auto [type]    = copy_mem_to_device->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_mem_to_device->arg(0));
        auto host_ptr    = emit(copy_mem_to_device->arg(1));
        auto dev_ptr     = emit(copy_mem_to_device->arg(2));
        auto size        = emit(w.lit_nat(Lit::as(type_size)));
        auto dev_ptr_i64 = bb.assign(name + "i64", "ptrtoint ptr addrspace(1) {} to i64", dev_ptr);

        auto copy_res
            = bb.assign(name + "res", "call i32 @cuMemcpyHtoD_v2(i64 {}, ptr {}, i64 {})", dev_ptr_i64, host_ptr, size);
        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto copy_mem_to_host = Axm::isa<gpu::copy_mem_to_host>(def)) {
        declare("i32 @cuMemcpyDtoH_v2(ptr, i64, i64)");

        auto [type]    = copy_mem_to_host->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_mem_to_host->arg(0));
        auto dev_ptr     = emit(copy_mem_to_host->arg(1));
        auto host_ptr    = emit(copy_mem_to_host->arg(2));
        auto size        = emit(w.lit_nat(Lit::as(type_size)));
        auto dev_ptr_i64 = bb.assign(name + "i64", "ptrtoint ptr addrspace(1) {} to i64", dev_ptr);

        auto copy_res
            = bb.assign(name + "res", "call i32 @cuMemcpyDtoH_v2(ptr {}, i64 {}, i64 {})", host_ptr, dev_ptr_i64, size);
        emit_cu_error_handling(bb, copy_res);
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

        auto lam = Lam::isa_mut_cn(launch->arg(3));
        if (!lam) error("kernel is not a lamda {}", func);
        if (!kernel_ids.contains(lam)) error("unknown kernel {}", lam);
        auto kid = kernel_ids[lam];

        auto func_ptr = bb.assign(name + "_funcptr", "alloca ptr");
        auto func_res = bb.assign(name + "_getfuncres", "call i32 @cuModuleGetFunction(ptr {}, ptr {}, ptr @.kname.{})",
                                  func_ptr, mod_name.value(), kid);
        emit_cu_error_handling(bb, func_res);

        auto args_ptr = bb.assign(name + "_args_ptr", "alloca ptr");
        print(bb.body().emplace_back(), "store {} {}, ptr {}", arg_type, arg, args_ptr);
        auto shared_mem_bytes = 0;      // TODO: add shared memory support
        auto stream           = "null"; // TODO: add support for CUDA streams
        auto launch_res       = bb.assign(name,
                                          "call i32 @cuLaunchKernel(ptr {}, i32 {}, i32 1, i32 1, i32 {}, i32 1, i32 1,"
                                                "i32 {}, ptr {}, ptr {}, ptr null)",
                                          func_ptr, n_warps, n_threads, shared_mem_bytes, stream, args_ptr);
        emit_cu_error_handling(bb, launch_res);
        return launch_res;
    }
    return std::nullopt;
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
            auto var     = root()->var(1);
            auto name    = id(var);
            locals_[var] = name;
            declare("i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
            bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
        }
        {
            // thread index
            auto var     = root()->var(2);
            auto name    = id(var);
            locals_[var] = name;
            declare("i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
            bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
        }
        {
            // TODO: remove, just for demonstration purposes
            // declare("i32 @vprintf(ptr, ptr)");
            // print(vars_decls_, "@welcome_message = private constant [13 x i8] c\"hi from t %d\\00\"\n");
            // auto thread_id = id(root()->var(2));
            // auto buf       = bb.assign("%buf", "alloca i32");
            // print(bb.body().emplace_back(), "store i32 {}, i32* {}", thread_id, buf);
            // print(bb.body().emplace_back(), "call i32 @vprintf(ptr @welcome_message, ptr {})", buf);
        }
    }

    return root()->unique_name();
}

std::optional<std::string> DeviceEmitter::isa_device_intrinsic(BB& bb, const Def* def) { return std::nullopt; }

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
