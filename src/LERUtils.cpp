// LERUtils.cpp
// ~~~~~~~~~~~~
// Implementation of more complex utility functions.
#include <ler-ir/LERUtils.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

using namespace llvm;
using namespace mlir;

namespace ler {

void moduleToExecutable(ModuleOp Module, StringRef Filename) {
  OUTS << "Generating executable.\n";
  registerBuiltinDialectTranslation(*Module.getContext());
  registerLLVMDialectTranslation(*Module.getContext());

  LLVMContext Ctx;

  auto LLVMIR = translateModuleToLLVMIR(Module.getOperation(), Ctx);

  std::error_code EC;
  raw_fd_stream LLVMOutFile("ler.ll", EC);
  LLVMIR->print(LLVMOutFile, nullptr);
}

} // namespace ler
