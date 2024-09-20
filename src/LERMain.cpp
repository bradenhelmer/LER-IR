// main.cpp
// ~~~~~~~~
// Entry into ler-opt tool.
#include <ler-ir/IR/LERDialect.h>
#include <ler-ir/LERCommonUtils.h>
#include <ler-ir/LERFrontend.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>

namespace cl = llvm::cl;
using namespace ler;

static cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"));

static cl::opt<bool> PrintAST("print-ast", cl::init(false));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  llvm::InitLLVM(argc, argv);
  OUTS << "Running ler-opt tool...\n";

  auto Parser =
      std::make_unique<LERParser>(std::make_unique<LERLexer>(InputFilename));

  LERStatement AST(Parser->getSourceRef());
  Parser->parseLERStatement(AST);

  if (PrintAST)
    AST.print();

  /*mlir::MLIRContext Context;*/
  /*mlir::OpBuilder Builder(&Context);*/
  /*auto Loop = Builder.create<RegularForLoopOp>(Builder.getUnknownLoc(), 1,
   * 100, 1);*/
  /*Loop.print(OUTS);*/

  return 0;
}
