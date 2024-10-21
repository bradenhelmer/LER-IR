// Misc.cpp
// ~~~~~~~~
// Miscellaneous analysis routine implementations.
#include <ler-ir/Analysis/Misc.h>
#include <llvm/ADT/StringMap.h>

using llvm::StringMap;

namespace {
static StringMap<BlockArgument> IdxToBlkArgMap;
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
