set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(LLVM_TARGET_DEFINITIONS LERDialect.td)

mlir_tablegen(LEROps.h.inc -gen-op-decls -dialect=ler)
mlir_tablegen(LEROps.cpp.inc -gen-op-defs -dialect=ler)

mlir_tablegen(LERDialect.h.inc -gen-dialect-decls -dialect=ler)
mlir_tablegen(LERDialect.cpp.inc -gen-dialect-defs -dialect=ler)

add_mlir_doc(LERDialect LERDialect Dialects/ -gen-dialect-doc)

add_public_tablegen_target(LERIRTableGen)
