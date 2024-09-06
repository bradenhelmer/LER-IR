// CommonUtils.h
// ~~~~~~~~~~~~~
// Common macro definitions and helper functions.
#ifndef LERIR_COMMON_UTILS_H
#define LERIR_COMMON_UTILS_H
#include <mlir/Dialect/Arith/IR/Arith.h>

#define OUTS llvm::outs()

static mlir::arith::ConstantOp QuickConstOp(mlir::OpBuilder &builder,
                                            uint64_t value,
                                            const std::string &name = "") {
  auto C = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      mlir::IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 64),
                             value));
  if (name != "")
    C->setAttr("sym_name",
               mlir::SymbolRefAttr::get(builder.getStringAttr(name)));
  return C;
}

#endif
