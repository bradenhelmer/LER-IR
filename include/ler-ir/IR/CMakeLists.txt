set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS LERBase.td)
mlir_tablegen(LERDialect.h.inc -gen-dialect-decls -dialect=ler)
mlir_tablegen(LERDialect.cpp.inc -gen-dialect-defs -dialect=ler)

set(LLVM_TARGET_DEFINITIONS LEROps.td)
mlir_tablegen(LEROps.h.inc -gen-ops-decls -dialect=ler)
mlir_tablegen(LEROps.cpp.inc -gen-ops-defs -dialect=ler)

add_public_tablegen_target(LERIRTableGen)
