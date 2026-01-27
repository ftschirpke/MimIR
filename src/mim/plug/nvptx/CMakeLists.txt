add_mim_plugin(nvptx
    SOURCES
        nvptx.cpp
        be/ll_nvptx.cpp
        ../core/be/ll.cpp # HACK: interim solution to link against LL emitter
    INSTALL
)
