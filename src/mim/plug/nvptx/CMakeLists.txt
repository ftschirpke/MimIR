add_mim_plugin(nvptx
    SOURCES
        nvptx.cpp
        be/ll_nvptx.cpp
        ../core/be/ll.cpp # HACK: interim solution to link against LL emitter
    INSTALL
)

# Make the generated header visible to the plugin's sources.
target_include_directories(mim_nvptx
    PRIVATE
        "${CMAKE_BINARY_DIR}/include"
)
