// InjectInductionVars.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~
// InjectInductionVars pass implementation.
#include <ler-ir/Analysis/Misc.h>
#include <ler-ir/LERUtils.h>
#include <ler-ir/Transforms/Passes.h>
#include <llvm/ADT/DenseMap.h>

using llvm::dyn_cast;
using llvm::SmallDenseMap;
using llvm::SmallVector;
using llvm::StringRef;
using mlir::BlockArgument;
using mlir::Operation;
using mlir::StringAttr;
using mlir::WalkOrder;
using mlir::WalkResult;
using namespace ler;

namespace ler {
#define GEN_PASS_DEF_INJECTINDUCTIONVARS
#include <ler-ir/Transforms/Passes.h.inc>
} // namespace ler

namespace {

struct InjectInductionVarsPass
    : public ler::impl::InjectInductionVarsBase<InjectInductionVarsPass> {
  using InjectInductionVarsBase::InjectInductionVarsBase;
  void runOnOperation() override {
    (void)getOperation().walk([&](Operation *Op) {
      BlockArgument BlkArg;
      if (auto VarOp = dyn_cast<VariableOp>(Op)) {
        if ((BlkArg = getBlkArgFromVarName(VarOp.getNameAsStrRef()))) {
          VarOp.replaceAllUsesWith(BlkArg);
          VarOp.erase();
        }
      }
      return WalkResult::advance();
    });
  }
};

} // namespace
