find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_subdirectory(bindings)

set(MIM_PY_VENV           "${CMAKE_BINARY_DIR}/.venv")
set(MIM_PY_PYTHON         "${MIM_PY_VENV}/bin/python")
set(MIM_PY_STAGE          "${CMAKE_BINARY_DIR}/mim_py_stage")
set(MIM_PY_WHEELS         "${CMAKE_BINARY_DIR}/mim_py_wheels")
set(MIM_PY_INSTALL_SCRIPT "${PROJECT_SOURCE_DIR}/cmake/MimPyInstallWheel.cmake")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import os, sys; print(os.path.realpath(getattr(sys, '_base_executable', sys.executable) or sys.executable))"
    OUTPUT_VARIABLE MIM_PY_BOOTSTRAP_PYTHON
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT EXISTS "${MIM_PY_BOOTSTRAP_PYTHON}")
    message(FATAL_ERROR
        "Could not determine a bootstrap Python interpreter from ${Python3_EXECUTABLE}"
    )
endif()

# venv: wiped & recreated on every build
add_custom_target(mim_py_venv
    COMMAND "${CMAKE_COMMAND}" -E echo "[mim_py] (re)creating venv at ${MIM_PY_VENV}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${MIM_PY_VENV}"
    COMMAND "${MIM_PY_BOOTSTRAP_PYTHON}" -m venv "${MIM_PY_VENV}"
    COMMAND "${MIM_PY_PYTHON}" -m pip install --upgrade pip
    COMMAND "${MIM_PY_PYTHON}" -m pip install build pybind11-stubgen
    COMMENT "mim_py: (re)creating ${MIM_PY_VENV}"
    VERBATIM
)

# Generate a stub for the internal pybind11 module.
set(_mim_py_stubs_build "${CMAKE_BINARY_DIR}/mim_py_stubs")
add_custom_target(mim_py_stubs
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_mim_py_stubs_build}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_mim_py_stubs_build}"
    COMMAND "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:pymim>"
            "${_mim_py_stubs_build}/$<TARGET_FILE_NAME:pymim>"
    COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONPATH=${_mim_py_stubs_build}"
            "${MIM_PY_PYTHON}" -m pybind11_stubgen
                --output-dir "${_mim_py_stubs_build}"
                _mim
    COMMENT "mim_py: generating _mim.pyi via pybind11-stubgen"
    VERBATIM
)
add_dependencies(mim_py_stubs pymim mim_py_venv)

# Main mim package: stage source, drop _mim.so plus generated enums/stubs in, build wheel, pip install.
set(_mim_py_main_stage  "${MIM_PY_STAGE}/main")
set(_mim_py_main_wheels "${MIM_PY_WHEELS}/main")
set(_mim_py_enum_stage  "${_mim_py_main_stage}/src/mim/_enums")
set(_mim_py_stage_enum_commands
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_mim_py_enum_stage}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_mim_py_enum_stage}/__init__.py"
)
foreach(_plugin ${MIM_PLUGIN_LIST})
    list(APPEND _mim_py_stage_enum_commands
        COMMAND "${CMAKE_COMMAND}" -E copy
                "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/mim/${_plugin}.py"
                "${_mim_py_enum_stage}/${_plugin}.py"
    )
endforeach()
add_custom_target(mim_py_main
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_mim_py_main_stage}" "${_mim_py_main_wheels}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}" "${_mim_py_main_stage}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf
            "${_mim_py_main_stage}/src/mim.egg-info"
    COMMAND "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:pymim>"
            "${_mim_py_main_stage}/src/mim/$<TARGET_FILE_NAME:pymim>"
    COMMAND "${CMAKE_COMMAND}" -E copy
            "${_mim_py_stubs_build}/_mim.pyi"
            "${_mim_py_main_stage}/src/mim/_mim.pyi"
    ${_mim_py_stage_enum_commands}
    COMMAND "${MIM_PY_PYTHON}" -m build --wheel
            --outdir "${_mim_py_main_wheels}" "${_mim_py_main_stage}"
    COMMAND "${CMAKE_COMMAND}"
            "-DPYTHON=${MIM_PY_PYTHON}"
            "-DWHEEL_DIR=${_mim_py_main_wheels}"
            -P "${MIM_PY_INSTALL_SCRIPT}"
    COMMENT "mim_py: building & installing main package"
    VERBATIM
)
add_dependencies(mim_py_main pymim mim_all_plugins mim_py_venv mim_py_stubs)

# Plugins: discovered at configure time under py/plugins/*/pyproject.toml.
file(GLOB _mim_py_plugin_tomls CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/plugins/*/pyproject.toml")
set(_mim_py_plugin_targets)
foreach(_toml ${_mim_py_plugin_tomls})
    get_filename_component(_plugin_dir  "${_toml}" DIRECTORY)
    get_filename_component(_plugin_name "${_plugin_dir}" NAME)
    set(_target mim_py_plugin_${_plugin_name})
    set(_stage   "${MIM_PY_STAGE}/plugins/${_plugin_name}")
    set(_wheels  "${MIM_PY_WHEELS}/plugins/${_plugin_name}")
    add_custom_target(${_target}
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_stage}" "${_wheels}"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${_plugin_dir}" "${_stage}"
        COMMAND "${MIM_PY_PYTHON}" -m build --wheel
                --outdir "${_wheels}" "${_stage}"
        COMMAND "${CMAKE_COMMAND}"
                "-DPYTHON=${MIM_PY_PYTHON}"
                "-DWHEEL_DIR=${_wheels}"
                -P "${MIM_PY_INSTALL_SCRIPT}"
        COMMENT "mim_py: building & installing ${_target}"
        VERBATIM
    )
    add_dependencies(${_target} mim_py_main)
    list(APPEND _mim_py_plugin_targets ${_target})
endforeach()

add_custom_target(mim_py ALL)
add_dependencies(mim_py mim_py_main ${_mim_py_plugin_targets})
