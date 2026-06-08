#include "mim/plug/pcuda/be/ll_pcuda.h"
#include "mim/plug/pcuda/be/hcf_adapter.h"
#include "mim/plug/pcuda/be/pcuda_config.h"

#include <iomanip>
#include <sstream>

#include <mim/driver.h>
#include <mim/util/sys.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/pcuda/pcuda.h>

using namespace std::string_literals;

namespace mim::ll::pcuda {

namespace core  = mim::plug::core;
namespace math  = mim::plug::math;
namespace mem   = mim::plug::mem;
namespace gpu   = mim::plug::gpu;
namespace pcuda = mim::plug::pcuda;

// ============================================================================
// PCUDAHostEmitter Class Definition
// ============================================================================

class PCUDAHostEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    PCUDAHostEmitter(World& world, std::ostream& ostream)
        : Super(world, "ll_pcuda_host_emitter", ostream) {}

    /// Provide the HCF bytes (output of HCFBuilder::serialize()) and the
    /// matching object-id. When set (non-empty hcf_blob), start() emits the
    /// stage-1 IR constants, AdaptiveCpp runtime externs, an llvm.global_ctors
    /// entry that registers the HCF at program start, and per-kernel name +
    /// storage globals used by the gpu::launch lowering.
    void set_hcf_embed(std::string hcf_blob, std::uint64_t object_id) {
        hcf_blob_  = std::move(hcf_blob);
        object_id_ = object_id;
    }

    void start() final;
    void find_kernels(const Def*);
    void emit_epilogue(Lam*) final;

    std::optional<std::string> isa_targetspecific_intrinsic(BB&, const Def*) final;

protected:
    std::string convert(const Def*) override;

private:
    void emit_hcf_embedding();

    LamMap<int> kernel_ids_;
    DefSet analyzed_;
    std::string hcf_blob_;        // serialized HCF bytes; empty = no embedding
    std::uint64_t object_id_ = 0; // matches @__acpp_local_sscp_hcf_object_id
};

// ============================================================================
// PCUDADeviceEmitter Class Definition
// ============================================================================

class PCUDADeviceEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    PCUDADeviceEmitter(World& world, std::ostream& ostream)
        : Super(world, "ll_pcuda_device_emitter", ostream) {}

    void start() final;
    std::string prepare() override;
    std::optional<std::string> isa_targetspecific_intrinsic(BB&, const Def*) final;

protected:
    std::string convert(const Def*) override;

private:
    LamSet kernels_;
    absl::btree_map<std::string, int> symbols_;
};

// ============================================================================
// PCUDAHostEmitter Implementation
// ============================================================================

void PCUDAHostEmitter::start() {
    for (auto def : world().annexes())
        find_kernels(def);
    for (auto def : world().externals().muts())
        find_kernels(def);

    if (!hcf_blob_.empty()) emit_hcf_embedding();

    Super::start();
}

namespace {

/// Build an LLVM byte-string literal `c"\XX\XX..."` for arbitrary bytes.
std::string llvm_byte_literal(std::string_view bytes) {
    std::ostringstream s;
    s << "c\"";
    s << std::hex << std::setfill('0');
    for (unsigned char c : bytes) s << "\\" << std::setw(2) << static_cast<int>(c);
    s << "\"";
    return s.str();
}

/// Build a NUL-terminated LLVM C-string literal for a kernel name.
std::string llvm_cstring_literal(std::string_view s) {
    std::ostringstream out;
    out << "c\"";
    out << std::hex << std::setfill('0');
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7F && c != '"' && c != '\\')
            out << static_cast<char>(c);
        else
            out << "\\" << std::setw(2) << static_cast<int>(c);
    }
    out << "\\00\"";
    return out.str();
}

} // namespace

void PCUDAHostEmitter::emit_hcf_embedding() {
    auto N = hcf_blob_.size();

    // Stage-1 IR constants (resolved by AdaptiveCpp's static_hcf_registration template).
    print(vars_decls_, "@__acpp_local_sscp_hcf_content = dso_local constant [{} x i8] {}\n", N,
          llvm_byte_literal(hcf_blob_));
    print(vars_decls_, "@__acpp_local_sscp_hcf_object_size = dso_local global i64 {}\n", N);
    print(vars_decls_, "@__acpp_local_sscp_hcf_object_id = dso_local global i64 {}\n", object_id_);

    // Per-kernel name + storage globals — referenced by gpu::launch lowering.
    for (auto [lam, kid] : kernel_ids_) {
        auto kname     = std::string{lam->sym().str()};
        auto kname_len = kname.size() + 1; // +1 for NUL
        print(vars_decls_, "@.kname_{} = private unnamed_addr constant [{} x i8] {}\n", kid, kname_len,
              llvm_cstring_literal(kname));
        print(vars_decls_, "@.kstorage_{} = internal global ptr null\n", kid);
    }

    // Runtime externs from libacpp-rt.so / libacpp-common.so.
    declare("void @__acpp_register_hcf(ptr, i64)");
    declare("void @__acpp_unregister_hcf(i64)");
    declare("void @__pcudaPushCallConfiguration(i64, i32, i64, i32, i64, ptr)");
    declare("i32 @__pcudaKernelCall(ptr, ptr, i64, ptr)");

    // Global constructor that registers the HCF blob at program startup.
    print(func_impls_,
          "define internal void @__mimir_acpp_register_hcf_ctor() {{\n"
          "  call void @__acpp_register_hcf(ptr @__acpp_local_sscp_hcf_content, i64 {})\n"
          "  ret void\n"
          "}}\n",
          N);
    print(vars_decls_,
          "@llvm.global_ctors = appending global [1 x {{ i32, ptr, ptr }}] "
          "[{{ i32, ptr, ptr }} {{ i32 65535, ptr @__mimir_acpp_register_hcf_ctor, ptr null }}]\n");
}

void PCUDAHostEmitter::find_kernels(const Def* def) {
    if (auto [_, ins] = analyzed_.emplace(def); !ins) return;

    for (auto d : def->deps())
        find_kernels(d);

    if (auto launch = Axm::isa<gpu::launch>(def)) {
        auto kernel     = launch->decurry()->decurry()->arg();
        auto kernel_lam = kernel->isa_mut<Lam>();
        assert(kernel_lam && "Expect kernel passed to %gpu.launch to be a mutable lambda");
        if (kernel_ids_.contains(kernel_lam)) return;
        auto kid                = kernel_ids_.size();
        kernel_ids_[kernel_lam] = kid;
    }
}

std::string PCUDAHostEmitter::convert(const Def* type) {
    if (auto ptr = Axm::isa<mem::Ptr>(type)) {
        auto [_, addr_space] = ptr->args<2>();
        auto lit             = Lit::isa(addr_space);
        if (lit.value_or(0L) != 0) {
            return "ptr";
        }
    } else if (auto symptr = Axm::isa<gpu::SymPtr>(type)) {
        auto [_, T, a]      = symptr->args<3>();
        auto ptr_equivalent = world().call<mem::Ptr>(Defs{T, a});
        return convert(ptr_equivalent);
    }
    return Super::convert(type);
}

void PCUDAHostEmitter::emit_epilogue(Lam* lam) {
    auto& bb = lam2bb_[lam];
    auto app = lam->body()->as<App>();
    if (auto ret = isa_targetspecific_intrinsic(bb, app)) {
        assert(ret.has_value());
        if (app->callee() == root()->ret_var())
            assert(false && "Return not implemented in pCUDA backend");
        else if (auto dispatch = Dispatch(app))
            assert(false && "Dispatch not implemented in pCUDA backend");
        else if (app->callee()->isa<Bot>())
            assert(false && "Bot not implemented in pCUDA backend");
        else if (auto _ = Lam::isa_mut_basicblock(app->callee()))
            assert(false && "Ordinary Jump not implemented in pCUDA backend");
        else if (Pi::isa_returning(app->callee_type()))
            bb.tail("br label {}", ret.value());
        else
            assert(false && "Unexpected return case in pCUDA backend");
    } else {
        Super::emit_epilogue(lam);
    }
}

// MimIR emits LLVM IR directly, so it must reference the *runtime symbols*
// exported by libacpp-rt.so, not the header-only template wrappers like
// `pcudaMalloc`/`pcudaMallocHost` (those are defined in
// AdaptiveCpp/include/hipSYCL/pcuda/pcuda_runtime.hpp and forward to the
// `pcudaAllocate*` runtime symbols below).
constexpr auto PCUDA_MALLOC          = "pcudaAllocateDevice";
constexpr auto PCUDA_FREE            = "pcudaFree";
constexpr auto PCUDA_MALLOC_HOST     = "pcudaAllocateHost";
constexpr auto PCUDA_FREE_HOST       = "pcudaFreeHost";
constexpr auto PCUDA_MEMCPY          = "pcudaMemcpy";
constexpr auto PCUDA_STREAM_CREATE   = "pcudaStreamCreate";
constexpr auto PCUDA_STREAM_DESTROY  = "pcudaStreamDestroy";
constexpr auto PCUDA_STREAM_SYNC     = "pcudaStreamSynchronize";
constexpr auto PCUDA_GET_DEVICE_PROP = "pcudaGetDeviceProperties";
constexpr auto PCUDA_SET_DEVICE      = "pcudaSetDevice";
constexpr auto PCUDA_GET_DEVICE      = "pcudaGetDevice";
constexpr auto PCUDA_GET_DEVICE_COUNT = "pcudaGetDeviceCount";

std::optional<std::string> PCUDAHostEmitter::isa_targetspecific_intrinsic(BB& bb, const Def* def) {
    auto name = id(def);

    if (auto default_stream = Axm::isa<gpu::default_stream>(def)) {
        return "null";
    } else if (auto init = Axm::isa<gpu::init>(def)) {
        declare("i32 @{}(ptr)", PCUDA_GET_DEVICE_COUNT);
        auto dev_count_ptr = bb.assign(name + "_devcount_ptr", "alloca i32");
        auto devcount_res = bb.assign(name + "_devcount_res",
                                      "call i32 @{}(ptr {})",
                                      PCUDA_GET_DEVICE_COUNT, dev_count_ptr);

        declare("i32 @{}(i32)", PCUDA_SET_DEVICE);
        auto setdev_res = bb.assign(name + "_setdev_res",
                                    "call i32 @{}(i32 0)",
                                    PCUDA_SET_DEVICE);

        auto [n, m, mem, global_syms_def, const_syms_def] = init->args<5>();
        return emit_unsafe(mem);
    } else if (auto deinit = Axm::isa<gpu::deinit>(def)) {
        emit_unsafe(deinit->arg(0));
        return emit_unsafe(deinit->arg(1));
    } else if (auto stream_init = Axm::isa<gpu::stream_init>(def)) {
        declare("i32 @{}(ptr)", PCUDA_STREAM_CREATE);
        emit_unsafe(stream_init->arg(0));
        emit_unsafe(stream_init->arg(1));
        auto stream_ptr = emit(stream_init->arg(2));
        auto res = bb.assign(name, "call i32 @{}(ptr {})",
                            PCUDA_STREAM_CREATE, stream_ptr);
        return res;
    } else if (auto stream_deinit = Axm::isa<gpu::stream_deinit>(def)) {
        // pcudaStream_t = pcuda::stream* — runtime ABI is a ptr, not i32.
        declare("i32 @{}(ptr)", PCUDA_STREAM_DESTROY);
        emit_unsafe(stream_deinit->arg(0));
        emit_unsafe(stream_deinit->arg(1));
        auto stream = emit(stream_deinit->arg(2));
        auto res = bb.assign(name, "call i32 @{}(ptr {})",
                            PCUDA_STREAM_DESTROY, stream);
        return res;
    } else if (auto stream_sync = Axm::isa<gpu::stream_sync>(def)) {
        declare("i32 @{}(ptr)", PCUDA_STREAM_SYNC);
        emit_unsafe(stream_sync->arg(0));
        emit_unsafe(stream_sync->arg(1));
        auto stream = emit(stream_sync->arg(2));
        auto res = bb.assign(name, "call i32 @{}(ptr {})",
                            PCUDA_STREAM_SYNC, stream);
        return res;
    } else if (auto alloc = Axm::isa<gpu::alloc>(def)) {
        declare("i32 @{}(ptr, i64)", PCUDA_MALLOC);
        emit_unsafe(alloc->arg(0));
        auto alloc_t    = alloc->decurry()->arg();
        World& w        = alloc_t->world();
        auto type_size  = w.call(core::trait::size, alloc_t);
        auto alloc_size = emit(type_size);
        auto ptr_t = convert(Axm::as<mem::Ptr>(def->proj(1)->type()));
        auto alloc_ptr = bb.assign(name + "ptr", "alloca {}", ptr_t);
        auto alloc_res = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {})",
                                   PCUDA_MALLOC, alloc_ptr, alloc_size);
        return bb.assign(name, "load {}, ptr {}", ptr_t, alloc_ptr);
    } else if (auto free = Axm::isa<gpu::free>(def)) {
        declare("i32 @{}(ptr)", PCUDA_FREE);
        emit_unsafe(free->arg(0));
        auto ptr = emit(free->arg(1));
        auto free_res = bb.assign(name + "res", "call i32 @{}(ptr {})",
                                 PCUDA_FREE, ptr);
        return free_res;
    } else if (auto copy_to_device = Axm::isa<gpu::copy_to_device>(def)) {
        declare("i32 @{}(ptr, ptr, i64, i32)", PCUDA_MEMCPY);
        auto type      = copy_to_device->decurry()->arg();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);
        emit_unsafe(copy_to_device->arg(0));
        emit_unsafe(copy_to_device->arg(1));
        auto host_ptr = emit(copy_to_device->arg(2));
        auto dev_ptr  = emit(copy_to_device->arg(3));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));
        auto copy_res = bb.assign(name + "res", "call i32 @{}(ptr {}, ptr {}, i64 {}, i32 1)",
                                 PCUDA_MEMCPY, dev_ptr, host_ptr, size);
        return copy_res;
    } else if (auto copy_to_host = Axm::isa<gpu::copy_to_host>(def)) {
        declare("i32 @{}(ptr, ptr, i64, i32)", PCUDA_MEMCPY);
        auto [type]    = copy_to_host->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);
        emit_unsafe(copy_to_host->arg(0));
        emit_unsafe(copy_to_host->arg(1));
        auto dev_ptr  = emit(copy_to_host->arg(2));
        auto host_ptr = emit(copy_to_host->arg(3));
        auto size     = emit(w.lit_nat(Lit::as(type_size)));
        auto copy_res = bb.assign(name + "res", "call i32 @{}(ptr {}, ptr {}, i64 {}, i32 2)",
                                 PCUDA_MEMCPY, host_ptr, dev_ptr, size);
        return copy_res;
    } else if (auto launch = Axm::isa<gpu::launch>(def)) {
        auto [implicits, launch_config, kernel_def, arg_def, func_args] = launch->uncurry_args<5>();
        auto [n_groups_def, n_items_def, stream_def, m, _, __, ___, MT] = launch_config->projs<8>();
        auto [mem, ret_lam_def] = func_args->projs<2>();

        Lam* lam = kernel_def->isa_mut<Lam>();
        if (!lam) error("kernel is not a lambda {}", kernel_def);
        if (!kernel_ids_.contains(lam)) error("unknown kernel {}", lam);

        // Host-only debug mode (no HCF blob set): consume operands but skip the
        // real launch ABI — the resulting LL is not runnable but is useful for
        // inspection. The full ABI requires emit_host_with_embedded_device.
        if (hcf_blob_.empty()) {
            emit_unsafe(mem);
            emit(n_groups_def);
            emit(n_items_def);
            emit(stream_def);
            return emit(ret_lam_def);
        }
        auto kid = kernel_ids_[lam];

        // dynamic shared memory bytes (literal at compile time per the launch_config)
        std::uint64_t shared_mem_bytes = 0;
        if (auto smem_count = Lit::as(m)) {
            if (smem_count != 1)
                error("You can only have one dynamic allocation of shared memory per kernel");
            shared_mem_bytes = Lit::as(world().call(core::trait::size, MT));
        }

        emit_unsafe(mem);
        auto n_groups = emit(n_groups_def);
        auto n_items  = emit(n_items_def);
        auto stream   = emit(stream_def);

        // Pack each dim3 as the x86-64 SysV coercion clang produces for libacpp-rt:
        //   dim3{x, y=1, z=1} -> (i64 xy = (1u64 << 32) | x, i32 z = 1)
        constexpr std::uint64_t y_eq_one_high = 1ull << 32;
        auto grid_x64 = bb.assign(name + "_grid_x64", "zext i32 {} to i64", n_groups);
        auto grid_xy  = bb.assign(name + "_grid_xy", "or i64 {}, {}", grid_x64, y_eq_one_high);
        auto blk_x64  = bb.assign(name + "_blk_x64", "zext i32 {} to i64", n_items);
        auto blk_xy   = bb.assign(name + "_blk_xy", "or i64 {}, {}", blk_x64, y_eq_one_high);

        print(bb.body().emplace_back(),
              "call void @__pcudaPushCallConfiguration(i64 {}, i32 1, i64 {}, i32 1, i64 {}, ptr {})",
              grid_xy, blk_xy, shared_mem_bytes, stream);

        // Flatten kernel args: one alloca + store per logical arg; pack pointers into [N x ptr].
        auto arg_arity = arg_def->num_projs();
        auto args_arr  = bb.assign(name + "_args", "alloca [{} x ptr]", arg_arity);
        for (std::size_t i = 0; i < arg_arity; ++i) {
            const Def* pi    = (arg_arity > 1) ? arg_def->proj(i) : arg_def;
            auto pi_val      = emit(pi);
            auto pi_type     = convert(pi->type());
            auto slot        = bb.assign(name + "_arg" + std::to_string(i) + "_slot",
                                         "alloca {}", pi_type);
            print(bb.body().emplace_back(), "store {} {}, ptr {}", pi_type, pi_val, slot);
            auto gep = bb.assign(name + "_arg" + std::to_string(i) + "_gep",
                                 "getelementptr [{} x ptr], ptr {}, i64 0, i64 {}",
                                 arg_arity, args_arr, i);
            print(bb.body().emplace_back(), "store ptr {}, ptr {}", slot, gep);
        }

        // Load the object-id value (NOT its address) — __pcudaKernelCall's third
        // arg is the runtime-known u64 used to key into hcf_cache.
        auto obj_id = bb.assign(name + "_objid", "load i64, ptr @__acpp_local_sscp_hcf_object_id");
        auto launch_res
            = bb.assign(name + "_kcall",
                        "call i32 @__pcudaKernelCall("
                        "ptr @.kname_{}, ptr {}, "
                        "i64 {}, "
                        "ptr @.kstorage_{})",
                        kid, args_arr, obj_id, kid);
        (void)launch_res;

        return emit(ret_lam_def);
    }

    return std::nullopt;
}

// ============================================================================
// PCUDADeviceEmitter Implementation
// ============================================================================

std::string PCUDADeviceEmitter::convert(const Def* type) {
    // Canonical SSCP device IR uses opaque `ptr` for all pointer types — no
    // explicit address space. The JIT's llvm-to-amdgpu / llvm-to-spirv passes
    // assign the right addrspace during backend flavoring. Leaving an explicit
    // addrspace(N) in the stage-1 IR causes the AMDGPU lowering to produce a
    // kernel that dereferences nil at runtime (GPU page fault).
    if (auto ptr = Axm::isa<mem::Ptr>(type)) return "ptr";
    if (auto symptr = Axm::isa<gpu::SymPtr>(type)) return "ptr";
    return Super::convert(type);
}

void PCUDADeviceEmitter::start() {
    for (auto kernel : world().externals().muts()) {
        auto kernel_lam = kernel->isa_mut<Lam>();
        assert(kernel_lam && "Expect kernel to be a mutable lambda");
        kernels_.emplace(kernel_lam);
    }
    Super::start();

    // SSCP kernel-discovery metadata: !hipsycl.sscp.annotations names each
    // kernel function and its dimensionality. Without this, AdaptiveCpp's
    // llvm-to-amdgpu pass cannot find the kernels in the embedded bitcode and
    // the JIT silently skips them.
    //
    // MimIR's GPU model has a single group_id + item_id per launch (1D), so
    // we always emit `hipsycl_kernel_dimension = 1`. Multi-dim launches would
    // need this to come from launch_config instead.
    std::ostringstream tmp;
    int next_id = 0;
    if (!kernels_.empty()) {
        std::vector<int> md_ids;
        for (auto k : kernels_) {
            md_ids.push_back(next_id);
            print(tmp, "!{} = !{{ptr {}, !\"hipsycl_kernel_dimension\", i32 1}}\n",
                  next_id, id(k));
            ++next_id;
        }
        print(tmp, "!hipsycl.sscp.annotations = !{{");
        const char* sep = "";
        for (auto i : md_ids) {
            print(tmp, "{}!{}", sep, i);
            sep = ", ";
        }
        print(tmp, "}}\n");
    }

    // AdaptiveCpp's PTX backend flavoring (LLVMToPtx::toBackendFlavor) appends
    // the nvvm-reflect ftz / prec-div / prec-sqrt settings via
    //   M.getModuleFlagsMetadata()->addOperand(...)
    // and that accessor returns null when the module carries no
    // !llvm.module.flags — so a device module without module flags makes the
    // JIT dereference null and segfault. A clang-generated SSCP module always
    // has module flags; MimIR's minimal device IR does not, so emit a benign
    // one to give the JIT a node to append to.
    int uwtable_id = next_id++;
    print(tmp, "!{} = !{{i32 1, !\"uwtable\", i32 0}}\n", uwtable_id);
    print(tmp, "!llvm.module.flags = !{{!{}}}\n", uwtable_id);

    ostream() << tmp.str();
}

std::string PCUDADeviceEmitter::prepare() {
    auto is_kern = kernels_.contains(root());
    if (!is_kern) return Super::prepare();
    auto kernel = root();

    // Generate generic kernel attributes compatible with SSCP/multiple backends
    // Instead of spir_kernel, use a generic function attribute
    print(func_impls_, "define {} {}(", convert_ret_pi(kernel->type()->ret_pi()), id(kernel));

    auto [m1, m3, m4, m5, group_id, item_id, symptrs, smem, arg, ret_lam] = kernel->vars<10>();

    auto arg_name     = id(arg);
    auto arg_type     = arg->type();
    auto arg_type_str = convert(arg_type);
    auto arg_arity    = arg->num_projs();
    locals_[arg]      = arg_name;

    auto& bb = lam2bb_[kernel];

    // SSCP "free kernel" ABI (FreeKernelCall.cpp:142-234): each logical kernel
    // arg gets its own LLVM parameter — the SSCP plugin's FreeKernelCallPass
    // generates a launcher that for each arg either passes the value directly
    // (pointer-to-ByVal-struct) or alloca's the value and passes &alloca, and
    // the runtime copies bytes per HCF parameter offsets/sizes. This matches
    // what MimIR already emits on the host side and what HCFBuilder records.
    if (arg_arity > 1) {
        // Tuple-typed arg: emit N flat LLVM params, synthesize the aggregate at
        // function entry via insertvalue chain so the body's existing
        // extractvalue uses continue to work unchanged.
        std::vector<std::pair<std::string, std::string>> flat_params;
        flat_params.reserve(arg_arity);
        const char* sep = "";
        for (size_t i = 0; i < arg_arity; ++i) {
            auto pt = convert(arg->proj(i)->type());
            auto pn = arg_name + ".f" + std::to_string(i);
            flat_params.emplace_back(pt, pn);
            print(func_impls_, "{}{} {}", sep, pt, pn);
            sep = ", ";
        }
        print(func_impls_, ") {{\n");

        std::string prev = "undef";
        for (size_t i = 0; i < arg_arity; ++i) {
            bool last      = (i + 1 == arg_arity);
            auto step_name = last ? arg_name : (arg_name + ".iv" + std::to_string(i));
            bb.assign(step_name, "insertvalue {} {}, {} {}, {}", arg_type_str, prev,
                      flat_params[i].first, flat_params[i].second, i);
            prev = step_name;
        }
    } else {
        // Single-arg kernel: one LLVM param matching the arg type. The body
        // uses arg_name directly (no extractvalue needed).
        print(func_impls_, "{} {}) {{\n", arg_type_str, arg_name);
    }

    auto register_sreg_idx = [&](const Def* def, std::string_view sreg) {
        auto name        = id(def);
        auto type        = def->type();
        auto type_name   = convert(type);
        auto opt_idx_lit = Idx::isa_lit(type);
        if (!opt_idx_lit) error("Type of '{}' must have known index type but has {}", def, type);
        auto idx_lit = opt_idx_lit.value();
        locals_[def] = name;

        std::string sscp_builtin;
        if (sreg == "ctaid.x")
            sscp_builtin = "__acpp_sscp_get_group_id";
        else if (sreg == "tid.x")
            sscp_builtin = "__acpp_sscp_get_local_id";
        else
            error("pCUDA backend: unknown work-item register '{}'", sreg);

        declare("i32 @{}(i32)", sscp_builtin);
        if (type_name == "i0") {
            locals_[def] = "0";
        } else if (type_name == "i32") {
            bb.assign(name, "call i32 @{}(i32 0)", sscp_builtin);
        } else if (idx_lit < (1u << 31)) {
            auto i32 = bb.assign(name + "i32", "call i32 @{}(i32 0)", sscp_builtin);
            bb.assign(name, "trunc i32 {} to {}", i32, type_name);
        } else {
            error("Work-item index type too large, must fit into I32");
        }
    };
    register_sreg_idx(group_id, "ctaid.x");
    register_sreg_idx(item_id, "tid.x");

    // Static/pre-declared shared memory: when the kernel's `smem` param is a
    // %gpu.SharedPtr T (not the empty-tuple default), emit a module-level
    // `addrspace(3) global T undef` and bind `locals_[smem]` to an
    // addrspacecast constexpr that yields a generic ptr. The body then uses
    // mem.lea/mem.store/etc. on the generic ptr unchanged.
    //
    // Canonical reference (acpp-built __shared__ var):
    //   @__acpp_local_mem.k.0 = internal unnamed_addr addrspace(3) global [4 x i32] undef
    //   ; uses: addrspacecast (ptr addrspace(3) @__acpp_local_mem.k.0 to ptr)
    if (auto smem_ptr = Axm::isa<mem::Ptr>(smem->type())) {
        auto [T, a] = smem_ptr->args<2>();
        auto a_lit  = Lit::as(a);
        auto gname  = "@" + smem->unique_name();
        print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", gname, a_lit, convert(T));
        locals_[smem] = fmt("addrspacecast (ptr addrspace({}) {} to ptr)", a_lit, gname);
    } else if (auto sigma = smem->type()->isa<Sigma>()) {
        assert(sigma->num_ops() == 0 && "Expect empty sigma for shared memory variable");
    }

    // Symbol pointers (gpu.GlobalSymPtr / gpu.ConstSymPtr): declared in the
    // device IR as externally-initialized globals with the appropriate
    // addrspace. The host-side runtime is expected to write to them via
    // gpu.symbol_copy_to_device — which on SSCP/pCUDA is not yet wired through
    // AdaptiveCpp's runtime, so kernels that USE these will JIT-link with
    // unresolved external symbols. The declaration alone keeps mim from
    // hitting the Pi-conversion assertion when the body references symptrs.
    auto bind_symptr = [&](const Def* var) {
        auto sym_t = Axm::isa<gpu::SymPtr>(var->type());
        if (!sym_t) return;
        auto [id_, T, a] = sym_t->args<3>();
        auto a_lit  = Lit::as(a);
        auto id_lit = Lit::as(id_);
        auto gname  = fmt("@__acpp_sscp_sym_a{}_{}", a_lit, id_lit);
        if (symbols_.emplace(gname, a_lit).second) {
            print(vars_decls_, "{} = dso_local addrspace({}) externally_initialized global {} undef\n",
                  gname, a_lit, convert(T));
        }
        locals_[var] = fmt("addrspacecast (ptr addrspace({}) {} to ptr)", a_lit, gname);
    };
    if (auto sigma = symptrs->type()->isa<Sigma>()) {
        if (sigma->num_ops() > 0) {
            // Multiple symbols — would need to bind each projection. Not
            // exercised by the current tests; deferred.
            error("pCUDA backend: kernel symptrs with multiple entries not yet supported");
        }
    } else if (Axm::isa<gpu::SymPtr>(symptrs->type())) {
        bind_symptr(symptrs);
    }

    return kernel->unique_name();
}

std::optional<std::string> PCUDADeviceEmitter::isa_targetspecific_intrinsic(BB& bb, const Def* def) {
    auto name = id(def);

    auto shared_as = Lit::as(world().annex<gpu::addr_space_shared>());

    // Shared-memory `mem.slot` (addrspace=shared): emit a module-level
    // addrspace(3) global and return an addrspacecast constexpr that yields a
    // generic ptr. The rest of the body uses normal mem.lea/load/store on the
    // generic ptr — LLVM verifies fine, and the AMDGPU JIT recognizes the
    // addrspace(3) global as local memory during backend flavoring.
    if (auto mslot = Axm::isa<mem::mslot>(def)) {
        auto [T, a] = mslot->decurry()->args<2>();
        auto a_lit  = Lit::as(a);
        if (a_lit == shared_as) {
            auto gname = "@" + def->unique_name();
            emit_unsafe(mslot->arg(0));
            print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", gname, a_lit, convert(T));
            return fmt("addrspacecast (ptr addrspace({}) {} to ptr)", a_lit, gname);
        }
        return std::nullopt;
    }

    // gpu::symptr2ptr — converts a gpu::SymPtr to a regular pointer. Since
    // our convert() returns "ptr" for both types, this is a no-op at the IR
    // level; just pass the inner pointer through.
    if (auto symptr2ptr = Axm::isa<gpu::symptr2ptr>(def)) {
        emit_unsafe(symptr2ptr->arg(0));
        return emit(symptr2ptr->arg(1));
    }

    // Override base-class `mem::malloc` handling when address-space != 0.
    // AMDGPU/SSCP has no usable device-side `malloc`/`free` symbol — the
    // AMDGPU linker errors out with `undefined symbol: malloc` if we leave
    // the call in. Lower per-thread heap-alloc to `alloca` (stack) instead.
    // Semantically equivalent for the typical "scratch per-thread" pattern
    // (e.g. scope_sync_kernel's `local_data`); the lifetime is bounded by
    // the kernel invocation which is exactly what `alloca` gives us.
    if (auto mlc = Axm::isa<mem::malloc>(def)) {
        auto as = mlc->decurry()->arg(1);
        if (Lit::as(as) != 0) {
            emit_unsafe(mlc->arg(0));
            // Element type comes from the result ptr's pointee.
            auto pointee = Axm::as<mem::Ptr>(def->proj(1)->type())->arg(0);
            return bb.assign(name, "alloca {}", convert(pointee));
        }
        return std::nullopt;
    } else if (auto fr = Axm::isa<mem::free>(def)) {
        auto as = fr->decurry()->arg(1);
        if (Lit::as(as) != 0) {
            // alloca is automatically released at function return; free is a no-op.
            emit_unsafe(fr->arg(0));
            emit_unsafe(fr->arg(1));
            return name;
        }
        return std::nullopt;
    }

    if (auto sync_work_items = Axm::isa<gpu::sync_work_items>(def)) {
        // Canonical SSCP barrier (AdaptiveCpp). Args: memory_scope, memory_order.
        // scope=2 (work_group), order=0 (relaxed) per hipSYCL/sycl/libkernel/memory.hpp.
        declare("void @__acpp_sscp_work_group_barrier(i32, i32)");
        emit_unsafe(sync_work_items->arg(0));
        emit_unsafe(sync_work_items->arg(1));
        print(bb.body().emplace_back(), "call void @__acpp_sscp_work_group_barrier(i32 2, i32 0)");
        return name;
    } else if (auto warp_size = Axm::isa<pcuda::warp_size>(def)) {
        // Use SSCP JIT reflection for warp size (backend-agnostic)
        declare("i32 @__acpp_sscp_jit_reflect_warp_size()");
        assert(name[0] == '%');
        auto valid_name = name.substr(1);
        bb.assign(valid_name, "call i32 @__acpp_sscp_jit_reflect_warp_size()");
        return valid_name;
    } else if (auto fmaf = Axm::isa<pcuda::fmaf>(def)) {
        // Fused multiply-add → SSCP builtin, picked by float width. The JIT
        // resolves __acpp_sscp_fma_f32/f64 against the kernel library for the
        // active backend, so this stays backend-agnostic.
        std::string_view builtin;
        std::string_view ty;
        switch (math::isa_f(fmaf->arg(0)->type()).value_or(0)) {
            case 32: builtin = "__acpp_sscp_fma_f32"; ty = "float"; break;
            case 64: builtin = "__acpp_sscp_fma_f64"; ty = "double"; break;
            default: error("pCUDA backend: %pcuda.fmaf only supports f32 and f64");
        }

        auto x = emit(fmaf->arg(0));
        auto y = emit(fmaf->arg(1));
        auto z = emit(fmaf->arg(2));

        declare("{} @{}({}, {}, {})", ty, builtin, ty, ty, ty);
        return bb.assign(name, "call {} @{}({} {}, {} {}, {} {})", ty, builtin, ty, x, ty, y, ty, z);
    }

    return std::nullopt;
}

// ============================================================================
// Backend emission functions
// ============================================================================

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

    PCUDAHostEmitter emitter(setup_phase->old_world(), ostream);
    emitter.run();
}

void emit_device(World& world, std::ostream& ostream) {
    auto [stage, setup_phase] = get_setup_stage(world);
    setup_phase->run();

    PCUDADeviceEmitter emitter(setup_phase->new_world(), ostream);
    emitter.run();
}

namespace {

/// Classify a MimIR type into one of the HCF parameter categories and return
/// its byte size. Pointers and SymPtrs are always 8 bytes; other types defer
/// to MimIR's `core::trait::size`.
std::pair<HCFParamType, std::size_t> classify_param(World& w, const Def* type) {
    if (Axm::isa<mem::Ptr>(type) || Axm::isa<gpu::SymPtr>(type))
        return {HCFParamType::Pointer, 8};

    auto size_def    = w.call(core::trait::size, type);
    std::size_t size = 0;
    if (auto sz = Lit::isa(size_def)) size = static_cast<std::size_t>(*sz);

    if (Idx::isa_lit(type)) return {HCFParamType::Integer, size};
    if (math::isa_f(type))  return {HCFParamType::FloatingPoint, size};
    return {HCFParamType::OtherByValue, size};
}

} // namespace

void emit_host_with_embedded_device(World& world, std::ostream& ostream) {
    static constexpr auto dev_ll_name   = "tmp_mimir_pcuda_dev.ll";
    static constexpr auto dev_bc_name   = "tmp_mimir_pcuda_dev.bc";
    static constexpr auto dev_link_name = "tmp_mimir_pcuda_dev_linked.bc";

    // SSCP stubs path is baked at configure time from the source tree by CMake.
    // If find_package(AdaptiveCpp) didn't run (MIM_WITH_ADAPTIVECPP=OFF), the
    // path is the empty string — fall back to cwd-relative for back-compat.
    std::string stub_path = config::SSCP_STUBS_PATH;
    if (stub_path.empty()) stub_path = "scripts/sscp_stubs.ll";

    auto [stage, setup_phase] = get_setup_stage(world);
    setup_phase->run();
    auto& new_w = setup_phase->new_world();

    // 1. Emit device LLVM IR to a temp file.
    {
        std::ofstream dev_ofs(dev_ll_name);
        if (!dev_ofs.is_open() || dev_ofs.fail())
            error("Cannot open temp file '{}' for device LL", dev_ll_name);
        PCUDADeviceEmitter device_emitter(new_w, dev_ofs);
        device_emitter.run();
    }

    // 2a. Convert .ll -> .bc via llvm-as.
    {
        auto llvm_as = sys::find_cmd("llvm-as");
        if (!std::filesystem::exists(llvm_as))
            error("Could not find command: llvm-as ({})", llvm_as);
        auto cmd = fmt("{} {} -o {}", llvm_as, dev_ll_name, dev_bc_name);
        auto rc  = sys::system(cmd);
        if (rc != 0) error("llvm-as exited with code {}", rc);
    }

    // 2b. Link in scripts/sscp_stubs.ll so the parameterized SSCP builtins
    // (__acpp_sscp_get_group_id / get_local_id) are defined inside the embedded
    // device bitcode. AdaptiveCpp's canonical kernel-library only ships the
    // per-dimension variants (`_x` / `_y` / `_z`); the stubs dispatch our
    // parameterized form to those. Without this link the SSCP JIT fails with
    // "undefined symbol: __acpp_sscp_get_group_id" at runtime on every backend.
    {
        auto llvm_link = sys::find_cmd("llvm-link");
        if (!std::filesystem::exists(llvm_link))
            error("Could not find command: llvm-link ({})", llvm_link);
        if (!std::filesystem::exists(stub_path))
            error("SSCP stubs file not found at '{}' "
                  "(run from project root or wire ACPP_INSTALL_DIR)", stub_path);
        auto cmd = fmt("{} {} {} -o {}", llvm_link, dev_bc_name, stub_path, dev_link_name);
        auto rc  = sys::system(cmd);
        if (rc != 0) error("llvm-link exited with code {}", rc);
    }

    // 3. Slurp the linked .bc bytes (post-llvm-link with sscp_stubs).
    std::string bc_bytes;
    {
        std::ifstream bc_ifs(dev_link_name, std::ios::binary);
        if (!bc_ifs) error("Cannot open generated bitcode '{}'", dev_link_name);
        std::ostringstream slurp;
        slurp << bc_ifs.rdbuf();
        bc_bytes = slurp.str();
    }

    // 4. Build the HCF blob (kernel metadata + bitcode attachment).
    HCFBuilder hcf;
    // TODO: derive object_id from a stable hash of the kernel set rather than a
    // fixed literal so that two TUs linked into one binary don't collide.
    constexpr std::uint64_t object_id = 0xACFFB1ECABCDEF01ull;
    hcf.set_object_id(object_id);
    hcf.set_device_bitcode(std::move(bc_bytes));

    std::vector<std::string> exported_symbols;
    for (auto def : new_w.externals().muts()) {
        auto lam = def->isa_mut<Lam>();
        if (!lam) continue;
        auto kname = std::string{lam->sym().str()};
        exported_symbols.push_back(kname);

        HCFKernel k;
        k.name = kname;

        // The 9th kernel param (index 8) is the user-data arg; the preceding
        // slots are mem effects (m1, m3, m4, m5), group_id, item_id, symptrs,
        // smem. The final slot is the return continuation.
        //
        // For **free kernels** (the MimIR model: each logical arg becomes one
        // LLVM parameter, not packed into a captures struct), each HCF
        // parameter has byte-offset = 0 within its own host-side `args[i]`
        // storage. Cf. acpp-built reference for `__global__ void k(int* a, int* b)`:
        // the HCF emits two parameters both at byte-offset 0, distinguished by
        // original-index 0 and 1. Cumulative offsets would mean SYCL/captures
        // semantics (one struct with multiple fields).
        auto arg   = lam->var(8);
        auto arity = arg->num_projs();
        for (std::size_t i = 0; i < arity; ++i) {
            const Def* pi    = (arity > 1) ? arg->proj(i) : arg;
            auto [cls, size] = classify_param(new_w, pi->type());
            k.host_side_parameter_sizes.push_back(size);
            k.parameters.push_back(HCFParam{0, size, i, cls, {}});
        }
        hcf.add_kernel(std::move(k));
    }
    hcf.set_exported_symbols(std::move(exported_symbols));

    auto hcf_blob = hcf.serialize();

    // 5. Run the host emitter, with the HCF blob to embed.
    PCUDAHostEmitter host_emitter(setup_phase->old_world(), ostream);
    host_emitter.set_hcf_embed(std::move(hcf_blob), object_id);
    host_emitter.run();

    // 6. Print the link recipe so users don't have to look it up. CMake baked
    // the AdaptiveCpp install dir into pcuda_config.h at configure time.
    if (config::ACPP_FOUND) {
        std::cerr << "[pcuda] emitted host module with embedded device bitcode.\n"
                     "[pcuda] link recipe:\n"
                     "[pcuda]   clang++ -O2 <host.ll> -o <app> \\\n"
                     "[pcuda]     -L"
                  << config::ACPP_LIB_DIR
                  << " -lacpp-rt -lacpp-common \\\n"
                     "[pcuda]     -Wl,-rpath="
                  << config::ACPP_LIB_DIR << " -pthread -ldl\n";
    }
}

} // namespace mim::ll::pcuda
