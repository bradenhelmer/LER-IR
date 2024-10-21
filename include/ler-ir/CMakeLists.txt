set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -dialect=ler)
add_public_tablegen_target(LERPassesIncTableGen)
add_subdirectory(IR)
