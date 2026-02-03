#include "mim/plug/nvptx/be/ll_nvptx.h"

#include <mim/driver.h>

#include <mim/util/sys.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/nvptx/nvptx.h>

using namespace std::string_literals;

namespace mim::ll::nvptx {

namespace mem  = mim::plug::mem;
namespace gpu  = mim::plug::gpu;
namespace core = mim::plug::core;

class HostEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    HostEmitter(World& world, std::ostream& ostream, std::optional<std::string> device_fatbin_file = std::nullopt)
        : Super(world, "llvm_nvptx_host_emitter", ostream)
        , device_fatbin_file_(device_fatbin_file) {}

    void start() final;
    void find_kernels(const Def*);

    void emit_epilogue(Lam*) final;

    std::optional<std::string> isa_targetspecific_intrinsic(BB&, const Def*) final;

protected:
    std::string convert(const Def*) override;

private:
    static constexpr std::string_view mod_name_          = "@mimir_cu_mod";
    static constexpr std::string_view ctx_name_          = "@mimir_cu_ctx";
    static constexpr std::string_view fatbin_name_       = "@fatbin";
    static constexpr std::string_view kernel_array_name_ = "@mimir_kernels";
    static constexpr std::string_view kernel_name_prefix = "@kname.";

    void emit_cu_error_handling(BB&, const std::string&, bool at_tail = false);

    std::optional<std::string> device_fatbin_file_;
    LamMap<int> kernel_ids_;

    DefSet analyzed_;
};

class DeviceEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    DeviceEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_device_emitter", ostream) {}

    void start() final;

    std::string prepare() override;

    std::optional<std::string> isa_targetspecific_intrinsic(BB&, const Def*) final;

private:
    LamSet kernels_;
};

void HostEmitter::start() {
    for (auto def : world().annexes())
        find_kernels(def);
    for (auto def : world().externals().muts())
        find_kernels(def);

    for (auto [kernel, kid] : kernel_ids_) {
        auto name = id(kernel).substr(1);
        print(vars_decls_, "{}{} = private constant [{} x i8] c\"{}\\00\"\n", kernel_name_prefix, kid, name.size() + 1,
              name);
    }
    print(vars_decls_, "{} = dso_local global [{} x ptr] zeroinitializer\n", kernel_array_name_, kernel_ids_.size());

    Super::start();
}

void HostEmitter::find_kernels(const Def* def) {
    if (auto [_, ins] = analyzed_.emplace(def); !ins) return;

    for (auto d : def->deps())
        find_kernels(d);

    if (auto launch = Axm::isa<gpu::launch>(def)) {
        auto kernel = launch->decurry()->arg();
        if (auto kernel_lam = kernel->isa_mut<Lam>()) {
            ILOG("FRIEDRICH HostEmitter thinks that '{}' is a kernel", kernel_lam);
            auto kid                = kernel_ids_.size();
            kernel_ids_[kernel_lam] = kid;
        }
    }
}

void HostEmitter::emit_cu_error_handling(BB& bb, const std::string& cu_result, bool tail) {
    // TODO: properly implement
#ifndef NDEBUG
    declare("i32 @cuGetErrorString(i32, ptr)");
    declare("i32 @puts(ptr)");

    auto err_name     = cu_result + "_errstr";
    auto err_name_ptr = cu_result + "_errname";

    if (tail) {
        bb.tail("{} = alloca ptr", err_name_ptr);
        // bb.tail("{}_issuccess = icmp eq i32 {}, 0", cu_result, cu_result);
        bb.tail("{}_errcall = call i32 @cuGetErrorString(i32 {}, ptr {})", cu_result, cu_result, err_name_ptr);
        bb.tail("{}_errstr = load ptr, ptr {}", cu_result, err_name_ptr);
        bb.tail("{}_errputs = call i32 @puts(ptr {})", cu_result, err_name);
    } else {
        bb.assign(err_name_ptr, "alloca ptr");
        // bb.assign(cu_result + "_issuccess", "icmp eq i32 {}, 0", cu_result);
        bb.assign(cu_result + "_errcall", "call i32 @cuGetErrorString(i32 {}, ptr {})", cu_result, err_name_ptr);
        bb.assign(err_name, "load ptr, ptr {}", err_name_ptr);
        bb.assign(cu_result + "_errputs", "call i32 @puts(ptr {})", err_name);
    }
#endif
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

void HostEmitter::emit_epilogue(Lam* lam) {
    auto& bb = lam2bb_[lam];

    // HACK: we partially re-implement the checks in Super::emit_epilogue to catch targetspecific applications
    auto app = lam->body()->as<App>();
    if (auto ret = isa_targetspecific_intrinsic(bb, app)) {
        assert(ret.has_value());
        if (app->callee() == root()->ret_var()) // return
            assert(false && "Return not implemented in NVPTX backend");
        else if (auto dispatch = Dispatch(app))
            assert(false && "Dispatch not implemented in NVPTX backend");
        else if (app->callee()->isa<Bot>())
            assert(false && "Bot not implemented in NVPTX backend");
        else if (auto callee = Lam::isa_mut_basicblock(app->callee())) // ordinary jump
            assert(false && "Ordinary Jump not implemented in NVPTX backend");
        else if (Pi::isa_returning(app->callee_type())) // function call
            bb.tail("br label {}", ret.value());
        else
            assert(false && "Unexpected return case in NVPTX backend");
    } else {
        Super::emit_epilogue(lam);
    }
}

std::optional<std::string> HostEmitter::isa_targetspecific_intrinsic(BB& bb, const Def* def) {
    auto name = id(def);
    std::string op;

    ILOG("FRIEDRICH HostEmitter def {} : {}", def, def->type());

    if (auto default_stream = Axm::isa<gpu::default_stream>(def)) {
        return "null";
    } else if (auto init = Axm::isa<gpu::init>(def)) {
        auto dev_num   = 0; // TODO: consider parameterizing this
        auto ctx_flags = 0; // TODO: consider parameterizing this

        declare("i32 @cuInit(i32)");
        auto init_res = bb.assign(name + "_init_res", "call i32 @cuInit(i32 0)");
        emit_cu_error_handling(bb, init_res);

        declare("i32 @cuDeviceGet(ptr, i32)");
        auto dev_ptr     = bb.assign(name + "_dev_ptr", "alloca i32");
        auto dev_get_res = bb.assign(name + "_get_res", "call i32 @cuDeviceGet(ptr {}, i32 {})", dev_ptr, dev_num);
        emit_cu_error_handling(bb, dev_get_res);

        declare("i32 @cuCtxCreate_v4(ptr, ptr, i32, i32)");
        print(vars_decls_, "{} = global ptr null\n", ctx_name_);
        auto dev     = bb.assign(name + "_dev", "load i32, ptr {}", dev_ptr);
        auto ctx_res = bb.assign(name + "_ctx_res", "call i32 @cuCtxCreate_v4(ptr {}, ptr null, i32 {}, i32 {})",
                                 ctx_name_, ctx_flags, dev);
        emit_cu_error_handling(bb, ctx_res);

        declare("i32 @cuModuleLoadFatBinary(ptr, ptr)");
        print(vars_decls_, "{} = global ptr null\n", mod_name_);
        if (device_fatbin_file_.has_value()) {
            std::ifstream fatbin_file(device_fatbin_file_.value(), std::ios::binary);
            if (!fatbin_file) error("Could not open {} as binary file", device_fatbin_file_.value());

            auto start = std::istreambuf_iterator<char>(fatbin_file);
            auto end   = std::istreambuf_iterator<char>();
            std::vector<u8> fatbin_bytes(start, end);

            print(vars_decls_, "{} = private constant [{} x i8] c\"", fatbin_name_, fatbin_bytes.size());
            for (auto byte : fatbin_bytes) {
                bool invalid_cstr_char = byte == '"' || byte == '\\';
                if (std::isprint(byte) && !invalid_cstr_char) {
                    print(vars_decls_, "{}", byte);
                } else {
                    auto byte_val = static_cast<int>(byte);
                    print(vars_decls_, "\\{x}{x}", byte_val / 16, byte_val % 16);
                }
            }
            print(vars_decls_, "\"\n");
        } else {
            print(vars_decls_, "; Add the bytes of your compiled nvptx fatbin binary here:\n");
            print(vars_decls_,
                  "{} = private constant [YOUR_FATBIN_DATA_SIZE_GOES_HERE x i8] [ YOUR_FATBIN_DATA_GOES_HERE ]\n",
                  fatbin_name_);
        }
        auto mod_res
            = bb.assign(name + "_mod_res", "call i32 @cuModuleLoadFatBinary(ptr {}, ptr {})", mod_name_, fatbin_name_);
        emit_cu_error_handling(bb, mod_res);
        auto mod_inner = bb.assign(name + "_mod_inner", "load ptr, ptr {}", mod_name_);

        declare("i32 @cuModuleGetFunction(ptr, ptr, ptr)");
        for (auto [kernel, kid] : kernel_ids_) {
            auto kname    = id(kernel).substr(1);
            auto func_ptr = bb.assign("%" + kname + "_funcptr", "getelementptr inbounds ptr, ptr {}, i64 {}",
                                      kernel_array_name_, kid);
            auto func_res
                = bb.assign("%" + kname + "_getfuncres", "call i32 @cuModuleGetFunction(ptr {}, ptr {}, ptr {}{})",
                            func_ptr, mod_inner, kernel_name_prefix, kid);
            emit_cu_error_handling(bb, func_res);
        }

        return emit_unsafe(init->arg());
    } else if (auto deinit = Axm::isa<gpu::deinit>(def)) {
        declare("i32 @cuModuleUnload(ptr)");
        bb.tail("{}_mod = load ptr, ptr {}", name, mod_name_);
        bb.tail("{}_mod_unload_res = call i32 @cuModuleUnload(ptr {}_mod)", name, name);
        emit_cu_error_handling(bb, name + "_mod_unload_res", true);

        declare("i32 @cuCtxDestroy_v2(ptr)");
        bb.tail("{}_ctx = load ptr, ptr {}", name, ctx_name_);
        bb.tail("{}_ctx_destroy_res = call i32 @cuCtxDestroy_v2(ptr {}_ctx)", name, name);
        emit_cu_error_handling(bb, name + "_ctx_destroy_res", true);

        emit_unsafe(deinit->arg(0));
        return emit_unsafe(deinit->arg(1));
    } else if (auto stream_init = Axm::isa<gpu::stream_init>(def)) {
        declare("i32 @cuStreamCreate(ptr, i32)");

        emit_unsafe(stream_init->arg(0));
        auto stream_ptr = emit(stream_init->arg(1));

        auto res = bb.assign(name, "call i32 @cuStreamCreate(ptr {}, i32 0)", stream_ptr);
        emit_cu_error_handling(bb, res);
        return res;
    } else if (auto stream_deinit = Axm::isa<gpu::stream_deinit>(def)) {
        declare("i32 @cuStreamDestroy(ptr)");

        emit_unsafe(stream_deinit->arg(0));
        auto stream_ptr = emit(stream_deinit->arg(1));
        auto stream     = bb.assign(name + "_inner", "load ptr, ptr {}", stream_ptr);

        auto res = bb.assign(name, "call i32 @cuStreamDestroy(ptr {})", stream);
        emit_cu_error_handling(bb, res);
        return res;
    } else if (auto stream_sync = Axm::isa<gpu::stream_sync>(def)) {
        declare("i32 @cuStreamSynchronize(ptr)");

        emit_unsafe(stream_sync->arg(0));
        auto stream_ptr = emit(stream_sync->arg(1));
        auto stream     = bb.assign(name + "_inner", "load ptr, ptr {}", stream_ptr);

        auto res = bb.assign(name, "call i32 @cuStreamSynchronize(ptr {})", stream);
        emit_cu_error_handling(bb, res);
        return res;
    } else if (auto alloc = Axm::isa<gpu::alloc>(def)) {
        bool is_async;
        switch (alloc.id()) {
            case gpu::alloc::block: is_async = false; break;
            case gpu::alloc::async: is_async = true; break;
            default: fe::unreachable();
        }

        if (is_async)
            declare("i32 @cuMemAllocAsync_v2(ptr, i64, ptr)");
        else
            declare("i32 @cuMemAlloc_v2(ptr, i64)");

        emit_unsafe(alloc->arg(0));
        auto type      = alloc->decurry()->arg();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        auto ptr_t = convert(Axm::as<mem::Ptr>(def->proj(1)->type()));

        auto alloc_ptr = bb.assign(name + "ptr", "alloca {}", ptr_t);
        std::string alloc_res;
        if (is_async) {
            auto stream = emit(alloc->arg(1));
            alloc_res   = bb.assign(name + "res", "call i32 @cuMemAllocAsync_v2(ptr {}, i64 {}, ptr {})", alloc_ptr,
                                    type_size, stream);
        } else
            alloc_res = bb.assign(name + "res", "call i32 @cuMemAlloc_v2(ptr {}, i64 {})", alloc_ptr, type_size);

        emit_cu_error_handling(bb, alloc_res);
        return bb.assign(name, "load {}, {} addrspace(0)* {}", ptr_t, ptr_t, alloc_ptr);
    } else if (auto free = Axm::isa<gpu::free>(def)) {
        bool is_async;
        switch (free.id()) {
            case gpu::free::block: is_async = false; break;
            case gpu::free::async: is_async = true; break;
            default: fe::unreachable();
        }

        if (is_async)
            declare("i32 @cuMemFreeAsync_v2(i64)");
        else
            declare("i32 @cuMemFree_v2(i64)");

        emit_unsafe(free->arg(0));
        auto ptr = emit(free->arg(1));

        std::string free_res;
        if (is_async) {
            auto stream = free->arg(2);
            free_res    = bb.assign(name + "res", "call i32 @cuMemFreeAsync_v2(i64 {}, ptr {})", ptr, stream);
        } else
            free_res = bb.assign(name + "res", "call i32 @cuMemFree_v2(i64 {})", ptr);

        emit_cu_error_handling(bb, free_res);
        return free_res;
    } else if (auto copy_to_device = Axm::isa<gpu::copy_to_device>(def)) {
        bool is_async;
        switch (copy_to_device.id()) {
            case gpu::copy_to_device::block: is_async = false; break;
            case gpu::copy_to_device::async: is_async = true; break;
            default: fe::unreachable();
        }

        if (is_async)
            declare("i32 @cuMemcpyHtoDAsync_v2(i64, ptr, i64, ptr)");
        else
            declare("i32 @cuMemcpyHtoD_v2(i64, ptr, i64)");

        auto type      = copy_to_device->decurry()->arg();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_device->arg(0));
        emit_unsafe(copy_to_device->arg(1));
        auto host_ptr = emit(copy_to_device->arg(2));
        auto dev_ptr  = emit(copy_to_device->arg(3));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));

        std::string copy_res;
        if (is_async) {
            auto stream = emit(copy_to_device->arg(4));
            copy_res    = bb.assign(name + "res", "call i32 @cuMemcpyHtoDAsync_v2(i64 {}, ptr {}, i64 {}, ptr {})",
                                    dev_ptr, host_ptr, size, stream);
        } else
            copy_res
                = bb.assign(name + "res", "call i32 @cuMemcpyHtoD_v2(i64 {}, ptr {}, i64 {})", dev_ptr, host_ptr, size);

        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto copy_to_host = Axm::isa<gpu::copy_to_host>(def)) {
        bool is_async;
        switch (copy_to_host.id()) {
            case gpu::copy_to_host::block: is_async = false; break;
            case gpu::copy_to_host::async: is_async = true; break;
            default: fe::unreachable();
        }
        if (is_async)
            declare("i32 @cuMemcpyDtoHAsync_v2(ptr, i64, i64, ptr)");
        else
            declare("i32 @cuMemcpyDtoH_v2(ptr, i64, i64)");

        auto [type]    = copy_to_host->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_host->arg(0));
        emit_unsafe(copy_to_host->arg(1));
        auto dev_ptr  = emit(copy_to_host->arg(2));
        auto host_ptr = emit(copy_to_host->arg(3));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));

        std::string copy_res;
        if (is_async) {
            auto stream = emit(copy_to_host->arg(4));
            copy_res    = bb.assign(name + "res", "call i32 @cuMemcpyDtoHAsync_v2(ptr {}, i64 {}, i64 {}, ptr {})",
                                    host_ptr, dev_ptr, size, stream);
        } else
            copy_res
                = bb.assign(name + "res", "call i32 @cuMemcpyDtoH_v2(ptr {}, i64 {}, i64 {})", host_ptr, dev_ptr, size);

        emit_cu_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto launch = Axm::isa<gpu::launch>(def)) {
        bool with_smem;
        switch (launch.id()) {
            case gpu::launch::no_smem: with_smem = false; break;
            case gpu::launch::with_smem: with_smem = true; break;
            default: fe::unreachable();
        }

        declare("i32 @cuLaunchKernel(ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr)");

        auto decurry1 = launch->decurry();
        auto decurry2 = decurry1->decurry();

        auto kernel_def = decurry1->arg();

        auto shared_mem_bytes = 0;
        if (with_smem) {
            auto def_smem_type = decurry2->arg(4);
            shared_mem_bytes   = Lit::as(world().call(core::trait::size, def_smem_type));
        }

        Lam* lam = kernel_def->isa_mut<Lam>();
        if (!lam) error("kernel is not a lamda {}", kernel_def);
        if (!kernel_ids_.contains(lam)) error("unknown kernel {}", lam);
        auto kid = kernel_ids_[lam];

        emit_unsafe(decurry2->arg(0));
        auto n_groups = emit(decurry2->arg(1));
        auto n_items  = emit(decurry2->arg(2));
        auto stream   = emit(decurry2->arg(3));
        auto kernel   = emit(kernel_def);
        auto arg      = emit(launch->arg(0));
        auto arg_type = convert(launch->arg(0)->type());
        auto ret_lam  = emit(launch->arg(1));

        auto func_ptr = bb.assign(name + "_kernptr", "getelementptr inbounds [{} x ptr], [{} x ptr]* {}, i64 0, i64 {}",
                                  kernel_ids_.size(), kernel_ids_.size(), kernel_array_name_, kid);
        auto func_inner = bb.assign(name + "_kernel", "load ptr, ptr {}", func_ptr);

        auto arg_wrap = bb.assign(name + "_arg_wrap", "alloca {}", arg_type);
        print(bb.body().emplace_back(), "store {} {}, ptr {}", arg_type, arg, arg_wrap);

        auto args_ptr = bb.assign(name + "_args_ptr", "alloca [1 x ptr]");
        print(bb.body().emplace_back(), "store ptr {}, ptr {}", arg_wrap, args_ptr);
        auto args_inner
            = bb.assign(name + "_args_inner", "getelementptr inbounds [1 x ptr], ptr {}, i64 0, i64 0", args_ptr);
        auto launch_res = bb.assign(name,
                                    "call i32 @cuLaunchKernel(ptr {}, i32 {}, i32 1, i32 1, i32 {}, i32 1, i32 1, "
                                    "i32 {}, ptr {}, ptr {}, ptr null)",
                                    func_inner, n_groups, n_items, shared_mem_bytes, stream, args_inner);
        emit_cu_error_handling(bb, launch_res);
        return ret_lam;
    }
    return std::nullopt;
}

void DeviceEmitter::start() {
    for (auto kernel : world().externals().muts())
        if (auto kernel_lam = kernel->isa_mut<Lam>()) {
            ILOG("FRIEDRICH DeviceEmitter thinks that '{}' is a kernel", kernel_lam);
            kernels_.emplace(kernel_lam);
        }
    Super::start();
}

std::string DeviceEmitter::prepare() {
    auto is_kern = kernels_.contains(root());
    if (!is_kern) return Super::prepare();
    auto kernel = root();

    print(func_impls_, "define ptx_kernel {} {}(", convert_ret_pi(kernel->type()->ret_pi()), id(kernel));

    auto num_vars = kernel->num_vars();
    assert(num_vars == 8 || num_vars == 9);

    auto groups_idx = kernel->var(4);
    auto items_idx  = kernel->var(5);

    const Def* smem_var = nullptr;
    if (num_vars == 9) smem_var = kernel->var(6);

    auto arg = kernel->var(num_vars - 2);
    {
        auto name    = id(arg);
        locals_[arg] = name;
        print(func_impls_, "{} {}) {{\n", convert(arg->type()), name);
    }

    auto& bb = lam2bb_[kernel];
    {
        auto name           = id(groups_idx);
        auto type           = groups_idx->type();
        auto type_name      = convert(type);
        auto idx_lit        = Idx::isa_lit(type);
        locals_[groups_idx] = name;
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
        auto name          = id(items_idx);
        auto type          = items_idx->type();
        auto type_name     = convert(type);
        auto idx_lit       = Idx::isa_lit(type);
        locals_[items_idx] = name;
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
    if (smem_var) {
        auto name         = "@" + smem_var->unique_name();
        locals_[smem_var] = name;
        auto ptr          = Axm::isa<mem::Ptr>(smem_var->type());
        assert(ptr);
        auto [T, a] = ptr->args<2>();
        print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", name, a, convert(T));
    }

    return kernel->unique_name();
}

std::optional<std::string> DeviceEmitter::isa_targetspecific_intrinsic(BB& bb, const Def* def) {
    auto name = id(def);
    std::string op;

    auto shared_as = Lit::as(world().annex<gpu::addr_space_shared>());

    ILOG("FRIEDRICH DeviceEmitter def {} : {}", def, def->type());

    if (auto mslot = Axm::isa<mem::mslot>(def)) {
        auto [T, a] = mslot->decurry()->args<2>();
        if (Lit::as(a) == shared_as) {
            name = "@" + def->unique_name();
            emit_unsafe(mslot->arg(0));
            print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", name, a, convert(T));
            return name;
        }
    } else if (auto sync_work_items = Axm::isa<gpu::sync_work_items>(def)) {
        declare("void @llvm.nvvm.barrier0()");

        emit_unsafe(sync_work_items->arg(0));
        emit_unsafe(sync_work_items->arg(1));
        print(bb.body().emplace_back(), "call void @llvm.nvvm.barrier0()");
        return name;
    }
    return std::nullopt;
}

static auto get_setup_stage(World& world) {
    auto flags         = Annex::base<gpu::setup4backend>();
    auto stage_funcptr = world.driver().stage(flags);
    auto stage         = (*stage_funcptr)(world);
    auto phase         = stage->isa<RWPhase>();
    if (!phase) error("Found unexpected gpu::setup4backend stage");
    return std::make_pair(std::move(stage), phase);
}

void emit_host(World& world, std::ostream& ostream) {
    auto [stage, setup_phase] = get_setup_stage(world);
    setup_phase->run();

    HostEmitter emitter(setup_phase->old_world(), ostream);
    emitter.run();
}

void emit_device(World& world, std::ostream& ostream) {
    auto [stage, setup_phase] = get_setup_stage(world);
    setup_phase->run();

    DeviceEmitter emitter(setup_phase->new_world(), ostream);
    emitter.run();
}

static std::optional<std::string> get_compute_capability() {
    auto out      = sys::exec("nvidia-smi --query-gpu=compute_cap");
    auto start    = out.find('\n') + 1;
    auto newline2 = out.find('\n', start);
    if (start < out.size()) out = out.substr(start, newline2 - start);
    // out should now have form "7.5" referencing the compute capability "sm_75"

    auto dot_pos = out.find('.');
    assert(dot_pos < out.size());

    for (size_t i = 0; i < out.size(); ++i)
        if (i != dot_pos && !std::isdigit(out[i])) return std::nullopt;

    return fmt("sm_{}{}", out.substr(0, dot_pos), out.substr(dot_pos + 1));
}

void emit_host_with_embedded_device(World& world, std::ostream& ostream) {
    static constexpr auto dev_ll_name     = "tmp_mimir_nvptx_dev.ll";
    static constexpr auto dev_ptx_name    = "tmp_mimir_nvptx_dev.ptx";
    static constexpr auto dev_cubin_name  = "tmp_mimir_nvptx_dev.cubin";
    static constexpr auto dev_fatbin_name = "tmp_mimir_nvptx_dev.fatbin";

    auto [stage, setup_phase] = get_setup_stage(world);
    setup_phase->run();

    {
        std::ofstream dev_ll_ofs;
        dev_ll_ofs.open(dev_ll_name);
        if (!dev_ll_ofs.is_open() || dev_ll_ofs.fail())
            error("Error occured while trying to open temporary file '{}'", dev_ll_name);
        DeviceEmitter device_emitter(setup_phase->new_world(), dev_ll_ofs);
        device_emitter.run();
    }

    auto compute_cap = get_compute_capability();
    std::string comp_cap;

    // TODO: find a way to pass on compute capability from CLI
    if (compute_cap.has_value()) {
        println(std::cout, "Determined compute capability to be '{}'", compute_cap.value());
        comp_cap = compute_cap.value();
    } else {
        static constexpr auto default_comp_cap = "sm_75";
        println(std::cout, "Could not determine compute capability, continuing with default: '{}'.", default_comp_cap);
        comp_cap = default_comp_cap;
    }
    {
        auto llc = sys::find_cmd("llc");
        if (!std::filesystem::exists(llc)) error("Could not find command: llc {}", llc);
        // TODO: support 32-bit version?
        auto cmd = fmt("{} -march=nvptx64 -mcpu={} {} -o {}", llc, comp_cap, dev_ll_name, dev_ptx_name);
        auto rc  = sys::system(cmd);
        if (rc != 0) {
            println(std::cout, "Command exited with error code {}", rc);
            return;
        }
    }
    {
        auto ptxas = sys::find_cmd("ptxas");
        if (!std::filesystem::exists(ptxas)) error("Could not find command: ptxas {}", ptxas);
        auto cmd = fmt("{} -arch={} {} -o {}", ptxas, comp_cap, dev_ptx_name, dev_cubin_name);
        auto rc  = sys::system(cmd);
        if (rc != 0) {
            println(std::cout, "Command exited with error code {}", rc);
            return;
        }
    }
    {
        auto nvcc = sys::find_cmd("nvcc");
        if (!std::filesystem::exists(nvcc)) error("Could not find command: nvcc {}", nvcc);
        auto cmd = fmt("{} -fatbin -arch={} {} -o {}", nvcc, comp_cap, dev_cubin_name, dev_fatbin_name);
        auto rc  = sys::system(cmd);
        if (rc != 0) {
            println(std::cout, "Command exited with error code {}", rc);
            return;
        }
    }

    HostEmitter host_emitter(setup_phase->old_world(), ostream, dev_fatbin_name);
    host_emitter.run();

#ifdef NDEBUG
    std::filesystem::remove(dev_ll_name);
    std::filesystem::remove(dev_ptx_name);
    std::filesystem::remove(dev_cubin_name);
    std::filesystem::remove(dev_fatbin_name);
#endif
}

} // namespace mim::ll::nvptx
