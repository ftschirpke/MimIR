add_mim_plugin(gpu
    SOURCES
        gpu.cpp
        phase/mem_checks.cpp
        phase/remove_double_syncs.cpp
    INSTALL
)
