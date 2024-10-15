// InjectInductionVars.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~
// InjectInductionVars pass implementation.

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
    SmallDenseMap<StringRef, BlockArgument> IdxToBlkArgMap;
    SmallVector<VariableOp, 32> ToBeErased;

    getOperation().walk<WalkOrder::PreOrder>([&](Operation *Op) {
      BlockArgument BlkArg;
      if (isForLoopOp(Op)) {
        auto IdxVarAttr = dyn_cast<StringAttr>(Op->getAttr("LoopIdxVar"));
        BlkArg = Op->getRegion(0).getArgument(0);
        IdxToBlkArgMap[IdxVarAttr.getValue()] = BlkArg;
      }
      if (auto VarOp = dyn_cast<VariableOp>(Op)) {
        BlkArg = IdxToBlkArgMap.lookup(VarOp.getNameAsStrRef());
        if (BlkArg) {
          ToBeErased.push_back(VarOp);
        }
      }
      return WalkResult::advance();
    });

    for (auto &E : ToBeErased) {
      auto BlkArg = IdxToBlkArgMap[E.getNameAsStrRef()];
      E.replaceAllUsesWith(BlkArg);
      E.erase();
    }
  }
};

} // namespace
