# Project Specific Information
cmake_minimum_required(VERSION 3.22)
project(LER-IR LANGUAGES CXX C)

# Misc CXX/CMake options.
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INCLUDE_CURRENT_DIR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fsanitize=address")

# Java Optimizer
set(OPTIMIZER_SRC_DIR ${CMAKE_SOURCE_DIR}/optimizer)
set(OPTIMIZER_BINARY_DIR ${CMAKE_BINARY_DIR}/optimizer)
add_custom_target(optimizer
	COMMAND
	${CMAKE_COMMAND} -E copy_directory 
	${OPTIMIZER_SRC_DIR}
	${OPTIMIZER_BINARY_DIR} &&
	javac "${OPTIMIZER_BINARY_DIR}/*.java"
)
add_custom_target(clean-optimizer
	COMMAND
	rm "${OPTIMIZER_BINARY_DIR}/*.class"
)

add_custom_target(ler-opt-script
	COMMAND
	cp "${CMAKE_SOURCE_DIR}/scripts/ler-full" "${CMAKE_BINARY_DIR}/" &&
	cp "${CMAKE_SOURCE_DIR}/scripts/notation_converter.py" "${CMAKE_BINARY_DIR}/"
)

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
set(LER_LIBS
    ${LLVM_LIBS}
    ${dialect_libs}
    LERAnalysis
    LERIR
    LERTransforms
    LERConversions
    MLIRToLLVMIRTranslationRegistration)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Executable
add_executable(ler-compile ${SRC})
add_dependencies(ler-compile LERAnalysis LERIR optimizer ler-opt-script)
target_link_libraries(ler-compile ${LER_LIBS})

add_subdirectory(include)
add_subdirectory(src)

add_custom_target(gen-docs
	COMMAND
	cp ${CMAKE_BINARY_DIR}/docs/Dialects/* ${CMAKE_SOURCE_DIR}/docs/
	DEPENDS
	mlir-doc
)
