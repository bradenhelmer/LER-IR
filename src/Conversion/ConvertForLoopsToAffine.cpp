// ConvertForLoopsToAffine.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ConvertForLoopsToAffine pass implementation.
#include <ler-ir/Conversion/Passes.h>

using mlir::Operation;
using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTFORLOOPSTOAFFINE
#include <ler-ir/Conversion/Passes.h.inc>
} // namespace ler

namespace {

struct ConvertForLoopsToAffinePass
    : public ler::impl::ConvertForLoopsToAffineBase<
          ConvertForLoopsToAffinePass> {
  using ConvertForLoopsToAffineBase::ConvertForLoopsToAffineBase;

  void runOnOperation() override {}
};
} // namespace
