# Install every wheel found in WHEEL_DIR into the venv whose interpreter is PYTHON.
# Invoked from CMakeLists.txt as: cmake -DPYTHON=<path> -DWHEEL_DIR=<path> -P <thisfile>
if(NOT PYTHON OR NOT WHEEL_DIR)
    message(FATAL_ERROR "MimPyInstallWheel: PYTHON and WHEEL_DIR are required")
endif()

file(GLOB _wheels "${WHEEL_DIR}/*.whl")
if(NOT _wheels)
    message(FATAL_ERROR "MimPyInstallWheel: no wheels found in ${WHEEL_DIR}")
endif()

execute_process(
    COMMAND "${PYTHON}" -m pip install --force-reinstall --no-deps ${_wheels}
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MimPyInstallWheel: pip install failed (exit ${_rc})")
endif()
