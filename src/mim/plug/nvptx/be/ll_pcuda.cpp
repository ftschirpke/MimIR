#include "mim/plug/nvptx/be/ll_pcuda.h"
#include "mim/plug/nvptx/be/hcf_adapter.h"

#include <mim/driver.h>
#include <mim/util/sys.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/nvptx/nvptx.h>

using namespace std::string_literals;

namespace mim::ll::pcuda {

namespace core  = mim::plug::core;
namespace math  = mim::plug::math;
namespace mem   = mim::plug::mem;
namespace gpu   = mim::plug::gpu;
namespace nvptx = mim::plug::nvptx;

// ============================================================================
// PCUDAHostEmitter Class Definition
// ============================================================================

class PCUDAHostEmitter : public mim::ll::Emitter {
public:
    using Super = mim::ll::Emitter;

    PCUDAHostEmitter(World& world, std::ostream& ostream)
        : Super(world, "ll_pcuda_host_emitter", ostream) {}

    void start() final;
    void find_kernels(const Def*);
    void emit_epilogue(Lam*) final;

    std::optional<std::string> isa_targetspecific_intrinsic(BB&, const Def*) final;

protected:
    std::string convert(const Def*) override;

private:
    LamMap<int> kernel_ids_;
    DefSet analyzed_;
    HCFMetadata hcf_metadata_;  // Track kernel metadata for HCF registration
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

    Super::start();
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

constexpr auto PCUDA_MALLOC          = "pcudaMalloc";
constexpr auto PCUDA_FREE            = "pcudaFree";
constexpr auto PCUDA_MALLOC_HOST     = "pcudaMallocHost";
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
        declare("i32 @{}(i32*)", PCUDA_GET_DEVICE_COUNT);
        auto dev_count_ptr = bb.assign(name + "_devcount_ptr", "alloca i32");
        auto devcount_res = bb.assign(name + "_devcount_res",
                                      "call i32 @{}(i32* {})",
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
        declare("i32 @{}(i32*)", PCUDA_STREAM_CREATE);
        emit_unsafe(stream_init->arg(0));
        emit_unsafe(stream_init->arg(1));
        auto stream_ptr = emit(stream_init->arg(2));
        auto res = bb.assign(name, "call i32 @{}(i32* {})",
                            PCUDA_STREAM_CREATE, stream_ptr);
        return res;
    } else if (auto stream_deinit = Axm::isa<gpu::stream_deinit>(def)) {
        declare("i32 @{}(i32)", PCUDA_STREAM_DESTROY);
        emit_unsafe(stream_deinit->arg(0));
        emit_unsafe(stream_deinit->arg(1));
        auto stream = emit(stream_deinit->arg(2));
        auto res = bb.assign(name, "call i32 @{}(i32 {})",
                            PCUDA_STREAM_DESTROY, stream);
        return res;
    } else if (auto stream_sync = Axm::isa<gpu::stream_sync>(def)) {
        declare("i32 @{}(i32)", PCUDA_STREAM_SYNC);
        emit_unsafe(stream_sync->arg(0));
        emit_unsafe(stream_sync->arg(1));
        auto stream = emit(stream_sync->arg(2));
        auto res = bb.assign(name, "call i32 @{}(i32 {})",
                            PCUDA_STREAM_SYNC, stream);
        return res;
    } else if (auto alloc = Axm::isa<gpu::alloc>(def)) {
        declare("i32 @{}(ptr*, i64)", PCUDA_MALLOC);
        emit_unsafe(alloc->arg(0));
        auto alloc_t    = alloc->decurry()->arg();
        World& w        = alloc_t->world();
        auto type_size  = w.call(core::trait::size, alloc_t);
        auto alloc_size = emit(type_size);
        auto ptr_t = convert(Axm::as<mem::Ptr>(def->proj(1)->type()));
        auto alloc_ptr = bb.assign(name + "ptr", "alloca {}", ptr_t);
        auto alloc_res = bb.assign(name + "res", "call i32 @{}(ptr* {}, i64 {})",
                                   PCUDA_MALLOC, alloc_ptr, alloc_size);
        return bb.assign(name, "load {}, ptr addrspace(0)* {}", ptr_t, alloc_ptr);
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

        emit_unsafe(mem);
        auto n_groups = emit(n_groups_def);
        auto n_items  = emit(n_items_def);
        auto stream   = emit(stream_def);
        auto kernel   = emit(kernel_def);
        auto arg      = emit(arg_def);
        auto ret_lam  = emit(ret_lam_def);

        return ret_lam;
    }

    return std::nullopt;
}

// ============================================================================
// PCUDADeviceEmitter Implementation
// ============================================================================

void PCUDADeviceEmitter::start() {
    for (auto kernel : world().externals().muts()) {
        auto kernel_lam = kernel->isa_mut<Lam>();
        assert(kernel_lam && "Expect kernel to be a mutable lambda");
        kernels_.emplace(kernel_lam);
    }
    Super::start();
}

std::string PCUDADeviceEmitter::prepare() {
    auto is_kern = kernels_.contains(root());
    if (!is_kern) return Super::prepare();
    auto kernel = root();

    print(func_impls_, "define spir_kernel {} {}(", convert_ret_pi(kernel->type()->ret_pi()), id(kernel));

    auto [m1, m3, m4, m5, group_id, item_id, symptrs, smem, arg, ret_lam] = kernel->vars<10>();

    auto arg_name = id(arg);
    locals_[arg]  = arg_name;
    print(func_impls_, "{} {}) {{\n", convert(arg->type()), arg_name);

    auto& bb = lam2bb_[kernel];

    auto register_sreg_idx = [&](const Def* def, std::string_view sreg) {
        auto name        = id(def);
        auto type        = def->type();
        auto type_name   = convert(type);
        auto opt_idx_lit = Idx::isa_lit(type);
        if (!opt_idx_lit) error("Type of '{}' must have known index type but has {}", def, type);
        locals_[def] = name;

        if (type_name == "i0") {
            locals_[def] = "0";
        } else if (type_name == "i32") {
            declare("i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
            declare("i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
            if (sreg == "ctaid.x")
                bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()");
            else if (sreg == "tid.x")
                bb.assign(name, "call i32 @llvm.nvvm.read.ptx.sreg.tid.x()");
        }
    };
    register_sreg_idx(group_id, "ctaid.x");
    register_sreg_idx(item_id, "tid.x");

    return kernel->unique_name();
}

std::optional<std::string> PCUDADeviceEmitter::isa_targetspecific_intrinsic(BB& bb, const Def* def) {
    auto name = id(def);

    if (auto sync_work_items = Axm::isa<gpu::sync_work_items>(def)) {
        declare("void @llvm.nvvm.barrier0()");
        emit_unsafe(sync_work_items->arg(0));
        emit_unsafe(sync_work_items->arg(1));
        print(bb.body().emplace_back(), "call void @llvm.nvvm.barrier0()");
        return name;
    } else if (auto warp_size = Axm::isa<nvptx::warp_size>(def)) {
        declare("i32 @llvm.nvvm.read.ptx.sreg.warpsize()");
        assert(name[0] == '%');
        auto valid_name = name.substr(1);
        bb.assign(valid_name, "call i32 @llvm.nvvm.read.ptx.sreg.warpsize()");
        return valid_name;
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

} // namespace mim::ll::pcuda
