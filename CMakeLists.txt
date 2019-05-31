cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)

project(compile-gcn)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited config" FORCE)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug or Release" FORCE)
endif()

find_package(LLVM REQUIRED)
add_definitions(${LLVM_DEFINITIONS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
llvm_map_components_to_libnames(LLVM_JIT_LIBRARIES all)

add_executable(compile_gcn compile_gcn.cpp log.h)
target_link_libraries(compile_gcn ${LLVM_JIT_LIBRARIES})
