// LERUtils.cpp
// ~~~~~~~~~~~~
// Implementation of more complex utility functions.
#include <filesystem>
#include <ler-ir/LERUtils.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>
#include <string>
#include <unistd.h>

using namespace llvm;
using namespace mlir;

namespace {
static inline bool llcExists() {
  std::string Cmd = "command -v llc > /dev/null 2>&1";
  return system(Cmd.c_str()) == 0;
}
} // namespace

namespace ler {

void moduleToExecutable(ModuleOp Module, StringRef Prefix) {
  OUTS << "Generating executable.\n";

  auto LLOutFile = (Prefix + ".ll").str();
  auto AsmOutFile = (Prefix + ".S").str();
  auto ExecOutFile = (Prefix + ".out").str();

  // Register translation interfaces
  registerBuiltinDialectTranslation(*Module.getContext());
  registerLLVMDialectTranslation(*Module.getContext());

  // Translate LLVMDialect to LLVM-IR
  LLVMContext Ctx;
  auto LLVMIR = translateModuleToLLVMIR(Module.getOperation(), Ctx);

  // Write to LLVM
  std::error_code EC;
  raw_fd_stream LLVMOutFile(LLOutFile, EC);
  LLVMIR->print(LLVMOutFile, nullptr);

  // Check LLC compiler exists on system.
  if (!llcExists()) {
    ERRS << "llc compiler does not exist on the system!";
    std::filesystem::remove(LLOutFile);
    exit(1);
  }

  // Construct compilation and assembly command.
  std::stringstream CmdStr;
  CmdStr << "llc " << LLOutFile << " -o " << AsmOutFile << " && "
         <<
#if defined(__clang__)
      "clang "
#elif defined(__GNUC__)
      "gcc "
#endif
         << AsmOutFile << " -o " << ExecOutFile;

  // Execute!
  system(CmdStr.str().c_str());

  // Remove intermediary files.
  /*std::filesystem::remove(LLOutFile);*/
  // std::filesystem::remove(AsmOutFile);

  return;
}

} // namespace ler
