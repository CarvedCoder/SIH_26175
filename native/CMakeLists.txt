cmake_minimum_required(VERSION 3.18)
project(dw_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  # Fall back to pybind11 fetched via pip (python -m pybind11 --cmakedir)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(PYBIND11_CMAKEDIR)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKEDIR}")
    find_package(pybind11 CONFIG REQUIRED)
  else()
    message(FATAL_ERROR "pybind11 not found: pip install pybind11")
  endif()
endif()

pybind11_add_module(dw_native src/bindings.cpp src/raster.cpp)

target_include_directories(dw_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# -O3 + native SIMD where the build host supports it; fall back to plain
# -O3 (baseline x86-64 still auto-vectorizes SSE2) when the flag is
# rejected by the compiler.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" DW_HAVE_MARCH_NATIVE)
if(DW_HAVE_MARCH_NATIVE)
  target_compile_options(dw_native PRIVATE -O3 -march=native)
else()
  if(MSVC)
    target_compile_options(dw_native PRIVATE /O2)
  else()
    target_compile_options(dw_native PRIVATE -O3)
  endif()
endif()

enable_testing()
add_executable(dw_raster_tests tests/test_dw_raster.cpp src/raster.cpp)
add_test(NAME raster_kernels COMMAND dw_raster_tests)
