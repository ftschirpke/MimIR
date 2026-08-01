// Runtime wrappers for the MimIR `ll_nvptx` backend.
//
// Like `ll`'s `mim_rt.c`, this file is compiled to textual LLVM IR by `clang` at build time (see
// `add_mim_runtime` in `cmake/Mim.cmake`) and embedded into or linked with the host module emitted
// by the `ll_nvptx` backend (`-X ll_nvptx:rt=embed|extern`); see issue #486.
//
// It intentionally does not include <cuda.h>: the wrappers only touch the CUresult return codes
// that the backend already has in hand, so they stay self-contained and build without the CUDA
// toolkit. Wrappers that need the driver API itself can be added here later.

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CU_CHECK(call)                                                                              \
    do {                                                                                            \
        CUresult res = call;                                                                        \
        if (res != CUDA_SUCCESS) {                                                                  \
            const char* err_string;                                                                 \
            cuGetErrorString(res, &err_string);                                                     \
            if (err_string) { fprintf(stderr, "Error: %s\ncaused by\n  " #call "\n", err_string); } \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

void mim_cu_init(CUcontext* cu_context,
                 CUmodule* cu_module,
                 CUfunction* kernel_ptrs,
                 const void* fatbinary,
                 const char** kernel_names,
                 size_t kernel_count) {
    // TODO: make hardcoded values parameterizable
    int device_num                = 0;
    CUctxCreateParams* ctx_params = NULL;
    int ctx_flags                 = 0;

    CU_CHECK(cuInit(0));
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_num));
    CU_CHECK(cuCtxCreate(cu_context, ctx_params, ctx_flags, cu_device));
    CU_CHECK(cuModuleLoadFatBinary(cu_module, fatbinary));

    for (size_t i = 0; i < kernel_count; ++i)
        CU_CHECK(cuModuleGetFunction(&kernel_ptrs[i], *cu_module, kernel_names[i]));

    // void* args[] = {&d_nums};
    // CHECK_CU(cuLaunchKernel(function, 1, 1, 1, THREAD_COUNT, 1, 1, 0, NULL, args, NULL));
}

void mim_cu_deinit(CUcontext* cu_context, CUmodule* cu_module) {
    CU_CHECK(cuModuleUnload(*cu_module));
    *cu_module = NULL;
    CU_CHECK(cuCtxDestroy(*cu_context));
    *cu_context = NULL;
}

void mim_cu_mem_alloc(CUdeviceptr* ptr, size_t bytesize) { CU_CHECK(cuMemAlloc(ptr, bytesize)); }

void mim_cu_mem_alloc_async(CUdeviceptr* ptr, size_t bytesize, CUstream stream) {
    CU_CHECK(cuMemAllocAsync(ptr, bytesize, stream));
}

void mim_cu_mem_free(CUdeviceptr ptr) { CU_CHECK(cuMemFree(ptr)); }

void mim_cu_mem_free_async(CUdeviceptr ptr, CUstream stream) { CU_CHECK(cuMemFreeAsync(ptr, stream)); }

void mim_cu_memcpy_htod(CUdeviceptr dst, const void* src, size_t bytesize) {
    CU_CHECK(cuMemcpyHtoD(dst, src, bytesize));
}

void mim_cu_memcpy_htod_async(CUdeviceptr dst, const void* src, size_t bytesize, CUstream stream) {
    CU_CHECK(cuMemcpyHtoDAsync(dst, src, bytesize, stream));
}

void mim_cu_memcpy_dtoh(void* dst, CUdeviceptr src, size_t bytesize) { CU_CHECK(cuMemcpyDtoH(dst, src, bytesize)); }

void mim_cu_memcpy_dtoh_async(void* dst, CUdeviceptr src, size_t bytesize, CUstream stream) {
    CU_CHECK(cuMemcpyDtoHAsync(dst, src, bytesize, stream));
}

void mim_cu_stream_create(CUstream* stream) { CU_CHECK(cuStreamCreate(stream, CU_STREAM_DEFAULT)); }

void mim_cu_stream_destroy(CUstream stream) { CU_CHECK(cuStreamDestroy(stream)); }

void mim_cu_stream_sync(CUstream stream) { CU_CHECK(cuStreamSynchronize(stream)); }

void mim_cu_launch_kernel(CUfunction kernel,
                          unsigned int grid_dim_x,
                          unsigned int block_dim_x,
                          unsigned int shared_mem_bytes,
                          CUstream stream,
                          void** kernel_params) {
    // TODO: consider cuLaunchKernelEx for modern launch attributes
    CU_CHECK(
        cuLaunchKernel(kernel, grid_dim_x, 1, 1, block_dim_x, 1, 1, shared_mem_bytes, stream, kernel_params, NULL));
}
