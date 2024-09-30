// LERDialect.cpp
// ~~~~~~~~~~~~~~
// Core LER dialect implementations.
#include <ler-ir/IR/LERDialect.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <ler-ir/IR/LERDialect.cpp.inc>

using namespace ler;
using llvm::ArrayRef;
using llvm::SmallVector;
using llvm::StringRef;
using mlir::Attribute;
using mlir::LogicalResult;
using mlir::OpBuilder;
using mlir::OperationState;
using mlir::Region;
using mlir::SymbolRefAttr;

void LERDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <ler-ir/IR/LEROps.cpp.inc>
      >();
}

namespace ler {
static std::pair<Attribute, Attribute>
convertForLoopBoundsToAttrs(OpBuilder &Builder, StringRef LBound,
                            StringRef UBound) {
  auto I64Type = Builder.getI64Type();
  Attribute LBoundAttr, UBoundAttr;
  uint64_t LBNum, UBNum;
  if (!LBound.getAsInteger(10, LBNum)) {
    LBoundAttr = Builder.getIntegerAttr(I64Type, LBNum);
  } else {
    LBoundAttr = SymbolRefAttr::get(Builder.getContext(), LBound);
  }
  if (!UBound.getAsInteger(10, UBNum)) {
    UBoundAttr = Builder.getIntegerAttr(I64Type, UBNum);
  } else {
    UBoundAttr = SymbolRefAttr::get(Builder.getContext(), UBound);
  }
  return std::make_pair(LBoundAttr, UBoundAttr);
}
} // namespace ler

#define GET_OP_CLASSES
#include <ler-ir/IR/LEROps.cpp.inc>

// ProductionForLoopOp
SmallVector<Region *> ProductionForLoopOp::getLoopRegions() {
  return {&getRegion()};
}

// RegularForLoopOp
SmallVector<Region *> RegularForLoopOp::getLoopRegions() {
  return {&getRegion()};
}

// SummationForLoopOp
SmallVector<Region *> SummationForLoopOp::getLoopRegions() {
  return {&getRegion()};
}

// WhileLoopOp
SmallVector<Region *> WhileLoopOp::getLoopRegions() { return {&getRegion()}; }
void WhileLoopOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        StringRef Condition, ArrayRef<std::string> Subscripts) {
  if (!Condition.empty())
    odsState.addAttribute("Condition", odsBuilder.getStringAttr(Condition));

  SmallVector<Attribute, 16> SubscriptAttrs;
  if (!Subscripts.empty()) {
    for (const auto &SS : Subscripts)
      SubscriptAttrs.push_back(odsBuilder.getStringAttr(SS));
  }
  odsState.addAttribute("Subscripts", odsBuilder.getArrayAttr(SubscriptAttrs));
  (void)odsState.addRegion();
}

/*LogicalResult ExpressionOp::verify() {}*/
