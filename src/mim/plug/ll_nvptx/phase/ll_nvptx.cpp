#include "mim/plug/ll_nvptx/phase/ll_nvptx.h"

#include <format>

#include <mim/driver.h>

#include <mim/util/sys.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/ll_nvptx/ll_nvptx.h>
#include <mim/plug/mem/mem.h>

using namespace std::string_literals;

namespace mim::plug::ll_nvptx {

namespace core = mim::plug::core;
namespace ll   = mim::plug::ll;
namespace mem  = mim::plug::mem;
namespace gpu  = mim::plug::gpu;

class HostEmitter : public ll::Emitter {
public:
    using Super = ll::Emitter;

    HostEmitter(World& world, std::ostream& ostream, std::optional<std::string> device_fatbin_file)
        : Super(world, "llvm_nvptx_host_emitter", ostream)
        , device_fatbin_file_(device_fatbin_file) {}

    void start() final;
    void find_kernels(const Def*);

    std::optional<std::string> isa_targetspecific_intrinsic(ll::BB&, const Def*) final;

protected:
    std::string convert(const Def*, bool simd = true) override;

private:
    static constexpr std::string_view cu_module_              = "@.mimir_cu_mod";
    static constexpr std::string_view cu_context_             = "@.mimir_cu_ctx";
    static constexpr std::string_view fatbin_data_            = "@.fatbin";
    static constexpr std::string_view kernel_name_prefix      = "@.kname.";
    static constexpr std::string_view kernel_names_array_     = "@.mimir_kernel_names";
    static constexpr std::string_view kernel_functions_array_ = "@.mimir_kernel_funcs";

    std::optional<std::string> device_fatbin_file_;
    LamMap<int> kernel_ids_;

    DefSet analyzed_;
};

class DeviceEmitter : public ll::Emitter {
public:
    using Super = ll::Emitter;

    DeviceEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_nvptx_device_emitter", ostream) {}

    void start() final;

    std::string prepare() override;

    std::optional<std::string> isa_targetspecific_intrinsic(ll::BB&, const Def*) final;

    bool is_using_libdevice() const { return uses_libdevice; }
    const std::string& get_extra_flags() const { return extra_flags; }

private:
    std::string convert(const Def* def, bool simd = false) override {
        if (simd) WLOG("Ignoring simd=true for type conversion in device code.");
        return Super::convert(def, false);
    }

    /// Device slots live in a module-scope global in their requested address space, not on the stack.
    std::string emit_slot(ll::BB&, const App* app, const Def* pointee, const Def* addr_space) override {
        auto v_ptr = "@" + app->unique_name() + ".slot";
        std::print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", v_ptr, addr_space, convert(pointee));
        return v_ptr;
    }

    absl::btree_map<std::string, int> symbols_;
    LamSet kernels_;

    bool uses_libdevice;
    std::string extra_flags;
};

void HostEmitter::start() {
    for (auto def : world().annexes().defs())
        find_kernels(def);
    for (auto def : world().externals().muts())
        find_kernels(def);

    std::stringstream kernel_names;
    std::print(kernel_names, "{} = dso_local global [{} x ptr] [", kernel_names_array_, kernel_ids_.size());
    auto sep = ""s;
    for (auto [kernel, kid] : kernel_ids_) {
        auto name = id(kernel).substr(1);
        std::print(vars_decls_, "{}{} = private constant [{} x i8] c\"{}\\00\"\n", kernel_name_prefix, kid,
                   name.size() + 1, name);
        std::print(kernel_names, "{}ptr getelementptr ([{} x i8], ptr {}{}, i64 0, i64 0)", sep, name.size(),
                   kernel_name_prefix, kid);
        sep = ", "s;
    }
    std::print(vars_decls_, "{}]\n", kernel_names.str());

    std::print(vars_decls_, "{} = dso_local global [{} x ptr] zeroinitializer\n", kernel_functions_array_,
               kernel_ids_.size());

    Super::start();
}

void HostEmitter::find_kernels(const Def* def) {
    if (auto [_, ins] = analyzed_.emplace(def); !ins) return;

    for (auto d : def->deps())
        find_kernels(d);

    if (auto launch = Axm::isa<gpu::launch>(def)) {
        auto kernel     = launch->decurry()->decurry()->arg();
        auto kernel_lam = kernel->expect_mut<Lam>("the kernel passed to %gpu.launch to be a mutable lambda");
        if (kernel_ids_.contains(kernel_lam)) return;
        auto kid                = kernel_ids_.size();
        kernel_ids_[kernel_lam] = kid;
    }
}

std::string HostEmitter::convert(const Def* type, bool simd) {
    if (auto ptr = Axm::isa<mem::Ptr>(type)) {
        auto [_, addr_space] = ptr->args<2>();
        auto lit             = Lit::isa(addr_space);
        if (lit.value_or(0L) != 0) {
            // NVIDIA treats all device pointers as i64s in host code
            return "i64";
        }
    }
    return Super::convert(type, simd);
}

std::optional<std::string> HostEmitter::isa_targetspecific_intrinsic(ll::BB& bb, const Def* def) {
    auto name = id(def);
    std::string op;

    if (auto default_stream = Axm::isa<gpu::default_stream>(def)) {
        return "null";
    } else if (auto init = Axm::isa<gpu::init>(def)) {
        declare_rt("void @mim_cu_init(ptr, ptr, ptr, ptr, ptr, i32)");

        std::print(vars_decls_, "{} = global ptr null\n", cu_context_);
        std::print(vars_decls_, "{} = global ptr null\n", cu_module_);

        if (device_fatbin_file_.has_value()) {
            std::ifstream fatbin_file(device_fatbin_file_.value(), std::ios::binary);
            if (!fatbin_file) fe::throwf("Could not open {} as binary file", device_fatbin_file_.value());

            auto start = std::istreambuf_iterator<char>(fatbin_file);
            auto end   = std::istreambuf_iterator<char>();
            std::vector<u8> fatbin_bytes(start, end);

            std::print(vars_decls_, "{} = private constant [{} x i8] c\"", fatbin_data_, fatbin_bytes.size());
            for (auto byte : fatbin_bytes) {
                bool invalid_cstr_char = byte == '"' || byte == '\\';
                if (std::isprint(byte) && !invalid_cstr_char) {
                    std::print(vars_decls_, "{:c}", byte);
                } else {
                    auto byte_val = static_cast<int>(byte);
                    std::print(vars_decls_, "\\{:x}{:x}", byte_val / 16, byte_val % 16);
                }
            }
            std::print(vars_decls_, "\"\n");
        } else {
            std::print(vars_decls_, "; Add the bytes of your compiled nvptx fatbin binary here:\n");
            std::print(vars_decls_,
                       "{} = private constant [YOUR_FATBIN_DATA_SIZE_GOES_HERE x i8] YOUR_FATBIN_DATA_GOES_HERE\n",
                       fatbin_data_);
        }

        std::print(bb.body().emplace_back(), "call void @mim_cu_init(ptr {}, ptr {}, ptr {}, ptr {}, ptr {}, i32 {})",
                   cu_context_, cu_module_, kernel_functions_array_, fatbin_data_, kernel_names_array_,
                   kernel_ids_.size());

        emit_unsafe(init->arg());
        return "";
    } else if (auto deinit = Axm::isa<gpu::deinit>(def)) {
        declare_rt("void @mim_cu_deinit(ptr, ptr)");

        emit_unsafe(deinit->arg(0));
        emit_unsafe(deinit->arg(1));
        emit_unsafe(deinit->arg(2));

        std::print(bb.body().emplace_back(), "call void @mim_cu_deinit(ptr {}, ptr {})", cu_context_, cu_module_);
        return "";
    } else if (auto stream_init = Axm::isa<gpu::stream_init>(def)) {
        declare_rt("void @mim_cu_stream_create(ptr)");

        emit_unsafe(stream_init->arg(0));
        emit_unsafe(stream_init->arg(1));
        auto stream_ptr = emit(stream_init->arg(2));

        std::print(bb.body().emplace_back(), "call void @mim_cu_stream_create(ptr {})", stream_ptr);
        return "";
    } else if (auto stream_deinit = Axm::isa<gpu::stream_deinit>(def)) {
        declare_rt("void @mim_cu_stream_destroy(ptr)");

        emit_unsafe(stream_deinit->arg(0));
        emit_unsafe(stream_deinit->arg(1));
        auto stream = emit(stream_deinit->arg(2));

        std::print(bb.body().emplace_back(), "call void @mim_cu_stream_destroy(ptr {})", stream);
        return "";
    } else if (auto stream_sync = Axm::isa<gpu::stream_sync>(def)) {
        declare_rt("void @mim_cu_stream_sync(ptr)");

        emit_unsafe(stream_sync->arg(0));
        emit_unsafe(stream_sync->arg(1));
        auto stream = emit(stream_sync->arg(2));

        std::print(bb.body().emplace_back(), "call void @mim_cu_stream_sync(ptr {})", stream);
        return "";
    } else if (auto alloc = Axm::isa<gpu::alloc>(def)) {
        bool is_async;
        switch (alloc.id()) {
            case gpu::alloc::block: is_async = false; break;
            case gpu::alloc::asyn: is_async = true; break;
            default: fe::throwf("ll_nvptx backend: unhandled %gpu.alloc id in '{}'", def);
        }

        if (is_async)
            declare_rt("void @mim_cu_mem_alloc_async(ptr, i64, ptr)");
        else
            declare_rt("void @mim_cu_mem_alloc(ptr, i64)");

        emit_unsafe(alloc->arg(0));
        auto alloc_t    = alloc->decurry()->arg();
        World& w        = alloc_t->world();
        auto type_size  = w.call(core::trait::size, alloc_t);
        auto alloc_size = emit(type_size);

        auto ptr_t = convert(Axm::expect<mem::Ptr>(def->proj(1)->type(), "a %mem.Ptr"));

        auto alloc_ptr = bb.assign(name + "ptr", "alloca {}", ptr_t);
        if (is_async) {
            auto stream = emit(alloc->arg(1));
            std::print(bb.body().emplace_back(), "call void @mim_cu_mem_alloc_async(ptr {}, i64 {}, ptr {})", alloc_ptr,
                       alloc_size, stream);
        } else
            std::print(bb.body().emplace_back(), "call void @mim_cu_mem_alloc(ptr {}, i64 {})", alloc_ptr, alloc_size);

        auto res = bb.assign(name, "load {}, {} addrspace(0)* {}", ptr_t, ptr_t, alloc_ptr);
        return res;
    } else if (auto free = Axm::isa<gpu::free>(def)) {
        bool is_async;
        switch (free.id()) {
            case gpu::free::block: is_async = false; break;
            case gpu::free::asyn: is_async = true; break;
            default: fe::throwf("ll_nvptx backend: unhandled %gpu.free id in '{}'", def);
        }

        if (is_async)
            declare_rt("void @mim_cu_mem_free_async(i64, ptr)");
        else
            declare_rt("void @mim_cu_mem_free(i64)");

        emit_unsafe(free->arg(0));
        auto ptr = emit(free->arg(1));

        if (is_async) {
            auto stream = emit(free->arg(2));
            std::print(bb.body().emplace_back(), "call void @mim_cu_mem_free_async(i64 {}, ptr {})", ptr, stream);
        } else
            std::print(bb.body().emplace_back(), "call void @mim_cu_mem_free(i64 {})", ptr);

        return "";
    } else if (auto copy_to_device = Axm::isa<gpu::copy_to_device>(def)) {
        bool is_async;
        switch (copy_to_device.id()) {
            case gpu::copy_to_device::block: is_async = false; break;
            case gpu::copy_to_device::asyn: is_async = true; break;
            default: fe::throwf("ll_nvptx backend: unhandled %gpu.copy_to_device id in '{}'", def);
        }

        if (is_async)
            declare_rt("void @mim_cu_memcpy_htod_async(i64, ptr, i64, ptr)");
        else
            declare_rt("void @mim_cu_memcpy_htod(i64, ptr, i64)");

        auto type      = copy_to_device->decurry()->arg();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_device->arg(0));
        emit_unsafe(copy_to_device->arg(1));
        auto host_ptr = emit(copy_to_device->arg(2));
        auto dev_ptr  = emit(copy_to_device->arg(3));
        auto size     = emit(type_size);

        if (is_async) {
            auto stream = emit(copy_to_device->arg(4));
            std::print(bb.body().emplace_back(), "call void @mim_cu_memcpy_htod_async(i64 {}, ptr {}, i64 {}, ptr {})",
                       dev_ptr, host_ptr, size, stream);
        } else
            std::print(bb.body().emplace_back(), "call void @mim_cu_memcpy_htod(i64 {}, ptr {}, i64 {})", dev_ptr,
                       host_ptr, size);

        return "";
    } else if (auto copy_to_host = Axm::isa<gpu::copy_to_host>(def)) {
        bool is_async;
        switch (copy_to_host.id()) {
            case gpu::copy_to_host::block: is_async = false; break;
            case gpu::copy_to_host::asyn: is_async = true; break;
            default: fe::throwf("ll_nvptx backend: unhandled %gpu.copy_to_host id in '{}'", def);
        }
        if (is_async)
            declare_rt("void @mim_cu_memcpy_dtoh_async(ptr, i64, i64, ptr)");
        else
            declare_rt("void @mim_cu_memcpy_dtoh(ptr, i64, i64)");

        auto [type]    = copy_to_host->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_host->arg(0));
        emit_unsafe(copy_to_host->arg(1));
        auto dev_ptr  = emit(copy_to_host->arg(2));
        auto host_ptr = emit(copy_to_host->arg(3));
        auto size     = emit(type_size);

        if (is_async) {
            auto stream = emit(copy_to_host->arg(4));
            std::print(bb.body().emplace_back(), "call void @mim_cu_memcpy_dtoh_async(ptr {}, i64 {}, i64 {}, ptr {})",
                       host_ptr, dev_ptr, size, stream);
        } else
            std::print(bb.body().emplace_back(), "call void @mim_cu_memcpy_dtoh(ptr {}, i64 {}, i64 {})", host_ptr,
                       dev_ptr, size);

        return "";
    } else if (auto launch = Axm::isa<gpu::launch>(def)) {
        declare_rt("void @mim_cu_launch_kernel(ptr, i32, i32, i32, ptr, ptr)");

        auto [implicits, launch_config, kernel_def, arg_def, func_args] = launch->uncurry_args<5>();
        auto [n_groups_def, n_items_def, stream_def, m, MT]             = launch_config->projs<5>();
        auto [mem, ret_lam_def]                                         = func_args->projs<2>();

        Lam* lam = kernel_def->isa_mut<Lam>();
        if (!lam) fe::throwf("kernel is not a lamda {}", kernel_def);
        if (!kernel_ids_.contains(lam)) fe::throwf("unknown kernel {}", lam);
        auto kid = kernel_ids_[lam];

        auto shared_mem_bytes = 0;
        if (auto smem_count = Lit::expect(m, "a shared-memory allocation count")) {
            if (smem_count != 1) fe::throwf("You can only have one dynamic allocation of shared memory per kernel");
            shared_mem_bytes = Lit::expect(world().call(core::trait::size, MT), "a shared-memory size");
        }

        emit_unsafe(mem);
        auto n_groups = emit(n_groups_def);
        auto n_items  = emit(n_items_def);
        auto stream   = emit(stream_def);
        auto kernel   = emit(kernel_def);
        auto arg      = emit(arg_def);
        auto arg_type = convert(arg_def->type());
        auto ret_lam  = emit(ret_lam_def);

        auto func_ptr = bb.assign(name + "_kernptr", "getelementptr inbounds [{} x ptr], [{} x ptr]* {}, i64 0, i64 {}",
                                  kernel_ids_.size(), kernel_ids_.size(), kernel_functions_array_, kid);
        auto func_inner = bb.assign(name + "_kernel", "load ptr, ptr {}", func_ptr);

        auto arg_wrap = bb.assign(name + "_arg_wrap", "alloca {}", arg_type);
        std::print(bb.body().emplace_back(), "store {} {}, ptr {}", arg_type, arg, arg_wrap);

        auto args_ptr = bb.assign(name + "_args_ptr", "alloca [1 x ptr]");
        std::print(bb.body().emplace_back(), "store ptr {}, ptr {}", arg_wrap, args_ptr);
        auto args_inner
            = bb.assign(name + "_args_inner", "getelementptr inbounds [1 x ptr], ptr {}, i64 0, i64 0", args_ptr);
        std::print(bb.body().emplace_back(),
                   "call void @mim_cu_launch_kernel(ptr {}, i32 {}, i32 {}, i32 {}, ptr {}, ptr {})", func_inner,
                   n_groups, n_items, shared_mem_bytes, stream, args_inner);
        return ret_lam;
    }
    return std::nullopt;
}

void DeviceEmitter::start() {
    for (auto kernel : world().externals().muts()) {
        auto kernel_lam = kernel->expect_mut<Lam>("an external kernel to be a mutable lambda");
        kernels_.emplace(kernel_lam);
    }
    Super::start();
    return;
}

std::string DeviceEmitter::prepare() {
    auto is_kern = kernels_.contains(root());
    if (!is_kern) return Super::prepare();
    auto kernel = root();

    std::print(func_impls_, "define ptx_kernel {} {}(", convert_ret_pi(kernel->type()->ret_pi()), id(kernel));

    auto [m1, m3, m4, m5, group_id, item_id, smem, arg, ret_lam] = kernel->vars<9>();

    auto arg_name = id(arg);
    locals_[arg]  = arg_name;
    std::print(func_impls_, "{} {}) {{\n", convert(arg->type()), arg_name);

    auto& bb = lam2bb_[kernel];

    auto register_sreg_idx = [&](const Def* def, std::string_view sreg) {
        auto name        = id(def);
        auto type        = def->type();
        auto type_name   = convert(type);
        auto opt_idx_lit = Idx::isa_lit(type);
        if (!opt_idx_lit) fe::throwf("Type of '{}' must have known index type but has {}", def, type);
        auto idx_lit = opt_idx_lit.value();
        locals_[def] = name;
        declare("i32 @llvm.nvvm.read.ptx.sreg.{}()", sreg);
        if (type_name == "i0") {
            locals_[def] = "0";
        } else if (type_name == "i32") {
            bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.{}()", sreg);
        } else if (idx_lit < (1u << 31)) {
            auto i32 = bb.assign(name + "i32", "call i32 @llvm.nvvm.read.ptx.sreg.{}()", sreg);
            bb.assign(name, "trunc i32 {} to {}", i32, type_name);
        } else {
            fe::throwf("Warp ID too large, must fit into I32");
        }
    };
    register_sreg_idx(group_id, "ctaid.x");
    register_sreg_idx(item_id, "tid.x");

    auto shared_as = Lit::expect(world().annex<gpu::addr_space_shared>(), "the shared address space");
    if (auto sigma = smem->type()->isa<Sigma>()) {
        if (sigma->num_ops() != 0)
            fe::throwf("ll_nvptx backend: shared-memory variable must be an empty sigma, but got '{}'", smem->type());
    } else {
        auto ptr    = Axm::expect<mem::Ptr>(smem->type(), "a shared-memory pointer type");
        auto [T, a] = ptr->args<2>();
        if (Lit::expect(a, "an address space") != shared_as)
            fe::throwf("ll_nvptx backend: shared-memory variable must live in the shared address space, but got '{}'",
                       smem->type());
        auto name     = "@" + smem->unique_name();
        locals_[smem] = name;
        std::print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", name, a, convert(T));
    }

    return kernel->unique_name();
}

std::optional<std::string> DeviceEmitter::isa_targetspecific_intrinsic(ll::BB& bb, const Def* def) {
    auto name = id(def);

    if (auto sync_work_items = Axm::isa<gpu::sync_work_items>(def)) {
        declare("void @llvm.nvvm.barrier0()");

        emit_unsafe(sync_work_items->arg(0));
        emit_unsafe(sync_work_items->arg(1));
        std::print(bb.body().emplace_back(), "call void @llvm.nvvm.barrier0()");
        return name;
    }
    return std::nullopt;
}

void emit_host(World& world, std::ostream& ostream, std::optional<std::string> device_fatbin_file, ll::Emitter::Rt rt) {
    HostEmitter emitter(world, ostream, device_fatbin_file);
    emitter.rt_mode(rt);
    // Same one-liner the `ll` backend uses; each backend just names its own runtime module.
    if (rt == ll::Emitter::Rt::embed) emitter.load_rt_module("ll_nvptx_rt.ll");
    emitter.run();
}

DeviceEmitFlags emit_device(World& world, std::ostream& ostream) {
    DeviceEmitter emitter(world, ostream);
    emitter.run();

    return DeviceEmitFlags{
        .uses_libdevice = emitter.is_using_libdevice(),
    };
}

} // namespace mim::plug::ll_nvptx
