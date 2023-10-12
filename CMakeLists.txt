cmake_minimum_required(VERSION 3.13.4 FATAL_ERROR)

project(compile-gcn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug or Release" FORCE)
endif()

add_executable(compile_gcn compile_gcn.cpp log.h)

#find_package(hsa-runtime64 PATHS /opt/rocm REQUIRED)
find_package(AMDDeviceLibs PATHS /opt/rocm REQUIRED)
get_target_property(ocml_LOCATION ocml LOCATION)
get_filename_component(HSA_BITCODE_PATH ${ocml_LOCATION} DIRECTORY)
get_filename_component(HSA_BITCODE_SUFFIX ${ocml_LOCATION} EXT)

find_package(LLVM REQUIRED)
if(LLVM_LINK_LLVM_DYLIB)
    set(LLVM_LINK_SHARED "USE_SHARED")
endif()
add_definitions(${LLVM_DEFINITIONS})
add_definitions(-DHSA_BITCODE_PATH="${HSA_BITCODE_PATH}" -DHSA_BITCODE_SUFFIX="${HSA_BITCODE_SUFFIX}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
llvm_config(compile_gcn ${LLVM_LINK_SHARED} irreader lto option support ${LLVM_TARGETS_TO_BUILD})
find_package(LLD REQUIRED)
target_link_libraries(compile_gcn PRIVATE lldELF lldCommon)
