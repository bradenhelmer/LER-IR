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
    SmallVector<VariableOp, 32> ToBeErased;

    (void)getOperation().walk([&](Operation *Op) {
      if (auto VarOp = dyn_cast<VariableOp>(Op)) {
        if (getBlkArgFromVarName(VarOp.getNameAsStrRef()))
          ToBeErased.push_back(VarOp);
      }
      return WalkResult::advance();
    });

    for (auto &E : ToBeErased) {
      auto BlkArg = getBlkArgFromVarName(E.getNameAsStrRef());
      E.replaceAllUsesWith(BlkArg);
      E.erase();
    }
  }
};

} // namespace
