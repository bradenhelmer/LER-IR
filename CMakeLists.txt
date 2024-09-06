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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

# Headers and Sources
file(GLOB SRC "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.h")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# External LLVM
find_package(LLVM REQUIRED CONFIG)
include(AddLLVM)
set(LLVM_LINK_COMPONENTS Core Option Support)
llvm_map_components_to_libnames(LLVM_LIBS ${LLVM_LINK_COMPONENTS})
include_directories(${LLVM_INCLUDE_DIRS})

# External MLIR
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include(TableGen)
set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MLIR_INCLUDE_DIRS})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Dependent libs
set(LER_LIBS ${LLVM_LIBS} ${dialect_libs} LERIR)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Executable
add_executable(ler-opt ${SRC})
add_dependencies(ler-opt LERIR)
target_link_libraries(ler-opt ${LER_LIBS})

# Misc Targets
add_custom_target(
  gen-tests
  COMMAND ler-opt -gen-tests="${CMAKE_SOURCE_DIR}"
  DEPENDS ler-opt)

add_subdirectory(docs)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
