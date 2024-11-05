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
using llvm::raw_fd_stream;
using mlir::PassManager;

// CLI arguments and options
cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                   cl::desc("<input file>"));

// Printing to stdout options
static cl::opt<bool> PrintAST("print-ast", cl::init(false));
static cl::opt<bool> PrintLERMLIR("print-ler-mlir", cl::init(false));
static cl::opt<bool> PrintLoweredMLIR("print-lowered-mlir", cl::init(false));

// MLIR out file options
static cl::opt<bool> OutputLERMLIR("output-ler-mlir", cl::init(false));
static cl::opt<bool> OutputLoweredMLIR("output-lowered-mlir", cl::init(false));

// Executable Options
static cl::opt<bool> CompileToExe("exe", cl::init(false));
cl::opt<bool> OutputLLVMIR("output-llvm-ir", cl::init(false));
cl::opt<bool> OutputAssembly("output-asm", cl::init(false));

static std::error_code EC;

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  llvm::InitLLVM(argc, argv);
  OUTS << "Running ler-opt tool...\n";

  auto RawFileName = InputFilename.substr(InputFilename.find_last_of('/') + 1);
  auto Prefix = RawFileName.substr(0, RawFileName.find_last_of('.'));
  StringRef MLIROutputPrefix(Prefix);

  // Parse into AST
  auto Parser =
      std::make_unique<LERParser>(std::make_unique<LERLexer>(InputFilename));

  LERStatement AST(Parser->getSourceRef());
  Parser->parseLERStatement(AST);

  if (PrintAST)
    AST.print();

  // Generate MLIR
  auto LERMLIR = AST.codeGen();

  if (PrintLERMLIR)
    LERMLIR.print(OUTS);

  if (OutputLERMLIR) {
    raw_fd_stream LERMLIROutFile((MLIROutputPrefix + "-ler.mlir").str(), EC);
    LERMLIR.print(LERMLIROutFile);
  }

  // Lower to LLVM Dialect
  PassManager PM = PassManager::on<ModuleOp>(LERMLIR.getContext());
  PM.addPass(createInjectInductionVars());
  PM.addPass(createConvertToArith());
  PM.addPass(createConvertArrayAccToMemref());
  PM.addPass(createConvertLoopsToAffineSCF());
  PM.addPass(createConvertToLLVM());

  if (failed(PM.run(LERMLIR)))
    LERMLIR.emitError("Pass error!");

  if (PrintLoweredMLIR)
    LERMLIR.print(OUTS);

  if (OutputLoweredMLIR) {
    raw_fd_stream LoweredMLIROutFile((MLIROutputPrefix + "-lowered.mlir").str(),
                                     EC);
    LERMLIR.print(LoweredMLIROutFile);
  }

  if (CompileToExe) {
    moduleToExecutable(LERMLIR, Prefix);
  } else {
    if (OutputLLVMIR)
      ERRS << "LLVM IR will not be generated and printed if -exe option not "
              "specified!\n";
    if (OutputAssembly)
      ERRS << "Assembly will not be generated and printed if -exe option not "
              "specified!\n";
  }

  return 0;
}
