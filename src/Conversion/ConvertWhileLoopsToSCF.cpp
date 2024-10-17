// ConvertWhileLoopsToSCF.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ConvertWhileLoopsToSCF pass implementation.
#include <ler-ir/Passes.h>

using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTWHILELOOPSTOSCF
#include <ler-ir/Passes.h.inc>
} // namespace ler

namespace {

struct ConvertWhileLoopsToSCFPass
    : public ler::impl::ConvertWhileLoopsToSCFBase<ConvertWhileLoopsToSCFPass> {
  using ConvertWhileLoopsToSCFBase::ConvertWhileLoopsToSCFBase;

  void runOnOperation() override {}
};
} // namespace
