// ConvertArrayAccToMemref.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ConvertArrayAccToMemref pass implementation.

#include <ler-ir/Conversion/Passes.h>

using mlir::Operation;
using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTARRAYACCTOMEMREF
#include <ler-ir/Conversion/Passes.h.inc>
} // namespace ler

namespace {

struct ConvertArrayAccToMemrefPass
    : public ler::impl::ConvertArrayAccToMemrefBase<
          ConvertArrayAccToMemrefPass> {
  using ConvertArrayAccToMemrefBase::ConvertArrayAccToMemrefBase;

  void runOnOperation() override {}
};
} // namespace
