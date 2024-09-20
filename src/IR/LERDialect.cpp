// LERDialect.cpp
// ~~~~~~~~~~~~~~
// Core LER dialect implementations.
#include <ler-ir/IR/LERDialect.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <ler-ir/IR/LERDialect.cpp.inc>
using namespace ler;
using llvm::SmallVector;
using mlir::Region;
void LERDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <ler-ir/IR/LEROps.cpp.inc>
      >();
}

#define GET_OP_CLASSES
#include <ler-ir/IR/LEROps.cpp.inc>

SmallVector<Region *> WhileLoopOp::getLoopRegions() {}

void RegularForLoopOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState,
                             int64_t LowerBound, int64_t UpperBound,
                             int64_t Step) {}
