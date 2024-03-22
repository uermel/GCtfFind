cmake_minimum_required(VERSION 3.23)
project(GCTFFind LANGUAGES C CXX CUDA)

# Set C++ 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set CUDA C++ 11
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set Flags for debug and release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g -m64")
set(CMAKE_CXX_FLAGS_RELEASE "-m64")

# Threads required
find_package( Threads REQUIRED )

# Output locations for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# When compiling python bindings, we need position independent code compilation (-fPIC)
message("skbuild: ${SKBUILD}")
if(SKBUILD)
    set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()

# Libraries
set(LIB_INC ${CMAKE_SOURCE_DIR}/inc/lib)
add_subdirectory(src/lib/CuUtil)
add_subdirectory(src/lib/CuUtilFFT)
add_subdirectory(src/lib/Mrcfile)
add_subdirectory(src/lib/Util)

# Executable
set(GCTFFIND_INC ${CMAKE_SOURCE_DIR}/inc/gctffind)
add_subdirectory(src/gctffind)

set(SKBUILD TRUE)

# Bindings
if(SKBUILD)
    message(STATUS "The project is built using scikit-build.")
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    # Add a library using FindPython's tooling (pybind11 also provides a helper like
    # this)
    set(PYGCTFFIND_INC ${CMAKE_SOURCE_DIR}/inc/pygctffind)
    add_subdirectory(src/pygctffind/_gctffind)
endif()

