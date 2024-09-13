// main.cpp
// ~~~~~~~~
// Entry into ler-opt tool.
#include <ler-ir/LERCommonUtils.h>
#include <ler-ir/LERFrontend.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

namespace cl = llvm::cl;
using namespace ler;

static cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  llvm::InitLLVM(argc, argv);
  OUTS << "Running ler-opt tool...\n";

  auto Parser =
      std::make_unique<LERParser>(std::make_unique<LERLexer>(InputFilename));

  Parser->lexAndPrintTokens();

  return 0;
}
