pybind11_add_module(pymim
    driver.cpp
    world.cpp
    sym.cpp
    py.cpp
    ast.cpp
    # app.cpp
    log.cpp
    flags.cpp
    lam.cpp
    parser.cpp
#    def_wrapper.cpp
    def.cpp
    error.cpp
)

set_target_properties(pymim PROPERTIES LIBRARY_OUTPUT_NAME _mim)
target_link_libraries(pymim PRIVATE libmim)
