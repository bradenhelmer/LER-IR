// InjectInductionVars.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~
// InjectInductionVars pass implementation.

#include <ler-ir/Transforms/Passes.h>

namespace ler {

#define GEN_PASS_DEF_INJECTINDUCTIONVARS
#include <ler-ir/Transforms/Passes.h.inc>

} // namespace ler
