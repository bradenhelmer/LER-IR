// main.cpp
// ~~~~~~~~
// Entry into ler-opt tool.
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/InitLLVM.h>

using namespace llvm;

int main(int argc, char **argv) {
  InitLLVM(argc, argv); 
  outs() << "Running ler-opt tool...\n";
}
