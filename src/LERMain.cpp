// LERMain.cpp
// ~~~~~~~~
// Entry into ler-opt tool.
#include <ler-ir/IR/LERDialect.h>
#include <ler-ir/LERFrontend.h>
#include <ler-ir/LERUtils.h>
#include <ler-ir/Passes.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/Pass/PassManager.h>

namespace cl = llvm::cl;
using namespace ler;
using mlir::PassManager;

// CLI arguments and options
cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                   cl::desc("<input file>"));
static cl::opt<bool> PrintAST("print-ast", cl::init(false));
static cl::opt<bool> PrintMLIR("print-mlir", cl::init(false));

static cl::opt<bool> CompileToExe("-exe", cl::init(false));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  llvm::InitLLVM(argc, argv);
  OUTS << "Running ler-opt tool...\n";

  // Parse into AST
  auto Parser =
      std::make_unique<LERParser>(std::make_unique<LERLexer>(InputFilename));

  LERStatement AST(Parser->getSourceRef());
  Parser->parseLERStatement(AST);

  if (PrintAST)
    AST.print();

  // Generate MLIR
  auto LERMLIR = AST.codeGen();

  // Lower to LLVM Dialect
  PassManager PM = PassManager::on<ModuleOp>(LERMLIR.getContext());
  PM.addPass(createInjectInductionVars());
  PM.addPass(createConvertToArith());
  PM.addPass(createConvertArrayAccToMemref());
  PM.addPass(createConvertLoopsToAffineSCF());
  PM.addPass(createConvertToLLVM());

  if (failed(PM.run(LERMLIR))) {
    LERMLIR.emitError("Pass error!");
  }

  if (CompileToExe) {
    auto RawFileName =
        InputFilename.substr(InputFilename.find_last_of('/') + 1);
    auto Prefix = RawFileName.substr(0, RawFileName.find_last_of('.'));
    moduleToExecutable(LERMLIR, Prefix);
  }

  return 0;
}
