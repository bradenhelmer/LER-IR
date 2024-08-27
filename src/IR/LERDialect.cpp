// LERDialect.cpp
// ~~~~~~~~~~~~~~
// Core LER dialect implementations.
#include <ler-ir/IR/LERDialect.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <ler-ir/IR/LERDialect.cpp.inc>
using namespace mlir::ler;
void LERDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <ler-ir/IR/LEROps.cpp.inc>
      >();
}
