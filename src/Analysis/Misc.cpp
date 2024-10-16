// Misc.cpp
// ~~~~~~~~
// Miscellaneous analysis routine implementations.
#include <ler-ir/Analysis/Misc.h>

using llvm::SmallDenseMap;

namespace {
SmallDenseMap<StringRef, BlockArgument> IdxToBlkArgMap;
} // namespace

namespace ler {

void insertIdxBlkArgMap(StringRef LoopIdxVar, BlockArgument BlkArg) {
  IdxToBlkArgMap[LoopIdxVar] = BlkArg;
}

BlockArgument getBlkArgFromVarName(StringRef Var) {
  if (IdxToBlkArgMap.find(Var) != IdxToBlkArgMap.end()) {
    return IdxToBlkArgMap[Var];
  }
  return nullptr;
}
} // namespace ler
