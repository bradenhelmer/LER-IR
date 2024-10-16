// Misc.h
// ~~~~~~
// Miscellaneous analysis routines.
#ifndef LERIR_ANALYSIS_MISC_H
#define LERIR_ANALYSIS_MISC_H
#include <mlir/IR/Value.h>

using llvm::StringRef;
using mlir::BlockArgument;

namespace ler {
void insertIdxBlkArgMap(StringRef LoopIdxVar, BlockArgument BlkArg);

BlockArgument getBlkArgFromVarName(StringRef Var);
} // namespace ler

#endif // LERIR_ANALYSIS_MISC_H
