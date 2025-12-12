add_mim_plugin(nvptx
    SOURCES
        nvptx.cpp
        normalizers.cpp
        be/ll_nvptx.cpp
        ../core/be/ll.cpp # HACK: not a good solution to the problem!
    INSTALL
)
