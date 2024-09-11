// main.cpp
// ~~~~~~~~
// Entry into ler-opt tool.
#include <ler-ir/CommonUtils.h>
#include <ler-ir/IR/LERDialect.h>
#include <llvm/Support/InitLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/MLIRContext.h>

using namespace ler;


int main(int argc, char **argv) {
  llvm::InitLLVM(argc, argv);
  llvm::outs() << "Running ler-opt tool...\n";

}
