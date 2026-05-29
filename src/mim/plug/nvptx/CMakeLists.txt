add_mim_plugin(nvptx
    SOURCES
        nvptx.cpp
        be/ll_nvptx.cpp
        be/ll_pcuda.cpp
        be/hcf_adapter.cpp
        ../core/be/ll.cpp # HACK: interim solution to link against LL emitter
    INSTALL
)

# pCUDA backend: bake AdaptiveCpp paths + scripts/sscp_stubs.ll location into
# a configured header so the orchestrator finds them regardless of cwd.
if(AdaptiveCpp_FOUND AND MIM_ACPP_INSTALL_DIR)
    set(MIM_ACPP_FOUND 1)
else()
    set(MIM_ACPP_FOUND 0)
    set(MIM_ACPP_INSTALL_DIR "")
    set(MIM_ACPP_LIB_DIR "")
endif()
set(MIM_SSCP_STUBS_PATH "${CMAKE_SOURCE_DIR}/scripts/sscp_stubs.ll")
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/be/pcuda_config.h.in"
    "${CMAKE_BINARY_DIR}/include/mim/plug/nvptx/be/pcuda_config.h"
    @ONLY
)
# Make the generated header visible to the plugin's sources.
target_include_directories(mim_nvptx
    PRIVATE
        "${CMAKE_BINARY_DIR}/include"
)
