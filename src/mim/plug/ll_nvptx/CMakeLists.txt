add_mim_plugin(ll_nvptx
    SOURCES
        ll_nvptx.cpp
        phase/ll_nvptx.cpp
    INSTALL
)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    set(MIM_CUDA_RT_ARGS "")
    foreach(dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
        list(APPEND MIM_CUDA_RT_ARGS "-I${dir}")
    endforeach()

    add_mim_runtime(ll_nvptx
        SOURCES
            rt/mim_cuda_rt.c
        INSTALL
        ARGS
            ${MIM_CUDA_RT_ARGS}
    )
else()
    message(STATUS "CUDA toolkit not found - skipping ll_nvptx's CUDA runtime (rt/mim_cuda_rt.c)")
endif()
