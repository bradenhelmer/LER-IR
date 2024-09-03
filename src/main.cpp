// main.cpp
// ~~~~~~~~
// Entry into ler-opt tool.
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/InitLLVM.h>


int main(int argc, char **argv) {
  llvm::InitLLVM(argc, argv);
  llvm::outs() << "Running ler-opt tool...\n";
}
