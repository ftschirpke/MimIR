cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(MimIR VERSION 0.1)

include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(CMakePrintHelpers)
include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(Mim)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20) # global standard for abseil
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

option(BUILD_SHARED_LIBS    "Build shared libraries." ON)
option(MIM_BUILD_DOCS       "If ON, build the documentation (requires Doxygen)." OFF)
option(MIM_BUILD_EXAMPLES   "If ON, build examples." OFF)
option(MIM_BUILD_PYTHON     "If ON, build the mim_py Python package (and any plugins) into a project-root .venv." OFF)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}; shared libs: ${BUILD_SHARED_LIBS}")

cmake_dependent_option(
    MIM_ENABLE_CHECKS
    "Enable expensive checks (only applicable in Debug mode)" ON
    "CMAKE_BUILD_TYPE STREQUAL Debug" OFF
)

if(WIN32)
    add_compile_definitions(NOMINMAX) # prevents windows.h defining min/max macros
else()
    add_compile_options(-Wall -Wextra)
endif()

add_custom_target(mim_all)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    find_program(MEMORYCHECK_COMMAND valgrind)
    set(VALGRIND_COMMAND_OPTIONS "--leak-check=full --error-exitcode=1")
    option(BUILD_TESTING "Build lit and unit tests." OFF) # needs to be BEFORE "include(CTest)"
    include(CTest)
    cmake_dependent_option(MIM_LIT_WITH_VALGRIND "If ON, the Mim CLI in the lit tests will be run under valgrind." OFF "BUILD_TESTING" OFF)
    set(MIM_LIT_TIMEOUT 20 CACHE STRING "Timout for lit tests.")
    if(BUILD_TESTING)
        include(GoogleTest)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
        add_subdirectory(gtest)
        add_subdirectory(lit)
    endif()
endif()

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_USE_EXTERNAL_GOOGLETEST ON)
set(ABSL_ENABLE_INSTALL ON)
set(FE_ABSL ON)

add_subdirectory(external/abseil-cpp)
add_subdirectory(external/pybind11)
add_subdirectory(external/fe)
add_subdirectory(external/lyra)
add_subdirectory(external/rang)
add_subdirectory(src)

if(MIM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(MIM_BUILD_DOCS)
    find_package(Doxygen REQUIRED dot)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs)
    endif()
endif()

# Build the `mim_py` Python package (plus any plugins under mim_py/plugins/*)
# and install them into a project-root .venv. Gated by MIM_BUILD_PYTHON;
# when ON, the `mim_py` umbrella target is part of the default build.
if(MIM_BUILD_PYTHON)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(MIM_PY_VENV           "${PROJECT_SOURCE_DIR}/.venv")
set(MIM_PY_PYTHON         "${MIM_PY_VENV}/bin/python")
set(MIM_PY_STAGE          "${CMAKE_BINARY_DIR}/mim_py_stage")
set(MIM_PY_WHEELS         "${CMAKE_BINARY_DIR}/mim_py_wheels")
set(MIM_PY_VENV_STAMP     "${CMAKE_BINARY_DIR}/.mim_py_venv.stamp")
set(MIM_PY_INSTALL_SCRIPT "${PROJECT_SOURCE_DIR}/cmake/MimPyInstallWheel.cmake")

# venv: create + install build backend (idempotent via stamp file)
add_custom_command(
    OUTPUT  "${MIM_PY_VENV_STAMP}"
    COMMAND "${CMAKE_COMMAND}" -E echo "[mim_py] preparing venv at ${MIM_PY_VENV}"
    COMMAND "${Python3_EXECUTABLE}" -m venv "${MIM_PY_VENV}"
    COMMAND "${MIM_PY_PYTHON}" -m pip install --upgrade pip
    COMMAND "${MIM_PY_PYTHON}" -m pip install build
    COMMAND "${CMAKE_COMMAND}" -E touch "${MIM_PY_VENV_STAMP}"
    COMMENT "mim_py: setting up ${MIM_PY_VENV}"
    VERBATIM
)
add_custom_target(mim_py_venv DEPENDS "${MIM_PY_VENV_STAMP}")

# main mim_py package: stage source, drop pymim .so in, build wheel, pip install
set(_mim_py_main_stage  "${MIM_PY_STAGE}/main")
set(_mim_py_main_wheels "${MIM_PY_WHEELS}/main")
set(_mim_py_main_stamp  "${CMAKE_BINARY_DIR}/.mim_py_main.stamp")
add_custom_command(
    OUTPUT  "${_mim_py_main_stamp}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_mim_py_main_stage}" "${_mim_py_main_wheels}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${PROJECT_SOURCE_DIR}/mim_py" "${_mim_py_main_stage}"
    COMMAND "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:pymim>"
            "${_mim_py_main_stage}/src/mim_py/$<TARGET_FILE_NAME:pymim>"
    COMMAND "${MIM_PY_PYTHON}" -m build --wheel
            --outdir "${_mim_py_main_wheels}" "${_mim_py_main_stage}"
    COMMAND "${CMAKE_COMMAND}"
            "-DPYTHON=${MIM_PY_PYTHON}"
            "-DWHEEL_DIR=${_mim_py_main_wheels}"
            -P "${MIM_PY_INSTALL_SCRIPT}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_mim_py_main_stamp}"
    DEPENDS pymim "${MIM_PY_VENV_STAMP}"
    COMMENT "mim_py: building & installing main package"
    VERBATIM
)
add_custom_target(mim_py_main DEPENDS "${_mim_py_main_stamp}")

# plugins: discovered at configure time under mim_py/plugins/*/pyproject.toml
file(GLOB _mim_py_plugin_tomls CONFIGURE_DEPENDS
     "${PROJECT_SOURCE_DIR}/mim_py/plugins/*/pyproject.toml")
set(_mim_py_plugin_targets)
foreach(_toml ${_mim_py_plugin_tomls})
    get_filename_component(_plugin_dir  "${_toml}" DIRECTORY)
    get_filename_component(_plugin_name "${_plugin_dir}" NAME)
    set(_target mim_py_plugin_${_plugin_name})
    set(_stage   "${MIM_PY_STAGE}/plugins/${_plugin_name}")
    set(_wheels  "${MIM_PY_WHEELS}/plugins/${_plugin_name}")
    set(_stamp   "${CMAKE_BINARY_DIR}/.${_target}.stamp")
    add_custom_command(
        OUTPUT  "${_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_stage}" "${_wheels}"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${_plugin_dir}" "${_stage}"
        COMMAND "${MIM_PY_PYTHON}" -m build --wheel
                --outdir "${_wheels}" "${_stage}"
        COMMAND "${CMAKE_COMMAND}"
                "-DPYTHON=${MIM_PY_PYTHON}"
                "-DWHEEL_DIR=${_wheels}"
                -P "${MIM_PY_INSTALL_SCRIPT}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
        DEPENDS "${_mim_py_main_stamp}" "${MIM_PY_VENV_STAMP}"
        COMMENT "mim_py: building & installing ${_target}"
        VERBATIM
    )
    add_custom_target(${_target} DEPENDS "${_stamp}")
    list(APPEND _mim_py_plugin_targets ${_target})
endforeach()

# umbrella target: part of default build because MIM_BUILD_PYTHON is ON
add_custom_target(mim_py ALL DEPENDS mim_py_main ${_mim_py_plugin_targets})
endif()

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mim-config-version.cmake"
    VERSION ${Mim_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mim-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mim-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mim"
)
install(
    TARGETS libmim
    EXPORT mim-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    FILES "${PROJECT_SOURCE_DIR}/external/half/include/half.hpp"
    DESTINATION include
)
install(
    FILES
        cmake/Mim.cmake
        "${CMAKE_CURRENT_BINARY_DIR}/mim-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mim-config-version.cmake"
    DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/mim
)
install(
    EXPORT mim-targets
    NAMESPACE mim::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mim
)
