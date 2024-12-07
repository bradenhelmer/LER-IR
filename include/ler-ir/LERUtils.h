// LERIR.h
// ~~~~~~~
// Commons
#ifndef LERIR_COMMON_UTILS_H
#define LERIR_COMMON_UTILS_H
#include <ler-ir/IR/LERDialect.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/MLIRContext.h>

using llvm::isa;
using mlir::MLIRContext;
using mlir::Operation;

namespace ler {

#define OUTS llvm::outs()
#define ERRS llvm::errs()

#define PRINT_INDENT()                                                         \
  for (int i = 0; i < Indent; ++i) {                                           \
    OUTS << "  ";                                                              \
  }

#define PRINT_SUBSCRIPTS()                                                     \
  if (Subscripts.size() > 0) {                                                 \
    OUTS << "Subscripts (";                                                    \
    auto SubscriptCount = getSubscriptCount();                                 \
    for (int i = 0; i < SubscriptCount; ++i) {                                 \
      OUTS << Subscripts[i];                                                   \
      if (i < SubscriptCount - 1) {                                            \
        OUTS << ',';                                                           \
      }                                                                        \
    }                                                                          \
    OUTS << ")";                                                               \
  }

static inline bool isForLoopOp(Operation *Op) {
  return isa<ProductionForLoopOp>(*Op) || isa<RegularForLoopOp>(*Op) ||
         isa<SummationForLoopOp>(*Op);
}

static inline bool isArithOp(Operation *Op) {
  return isa<AddOp>(*Op) || isa<SubOp>(*Op) || isa<MulOp>(*Op) ||
         isa<DivOp>(*Op);
}

static inline bool isConstantOp(Operation *Op) { return isa<ConstantOp>(*Op); }

void moduleToExecutable(mlir::ModuleOp, llvm::StringRef);

} // namespace ler

#endif
