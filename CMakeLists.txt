# Project Specific Information
cmake_minimum_required(VERSION 3.25)
project(LER-IR LANGUAGES CXX C)

# Misc CXX/CMake options.
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INCLUDE_CURRENT_DIR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Debug")

# Headers and Sources
file(GLOB SRC "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.h")
include_directories("${CMAKE_SOURCE_DIR}/include")

# External
find_package(LLVM REQUIRED CONFIG)
set(LLVM_LINK_COMPONENTS Core Support IRReader IRPrinter)
llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_LINK_COMPONENTS})

# Executable
add_executable(ler-opt ${SRC})
target_link_libraries(ler-opt ${LLVM_LIBS})

add_subdirectory(include)
add_subdirectory(src)
