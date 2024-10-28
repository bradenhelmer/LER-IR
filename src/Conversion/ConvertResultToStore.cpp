// ConvertResultToStore.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
#include <ler-ir/Passes.h>

using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTRESULTTOSTORE
#include <ler-ir/Passes.h.inc>
} // namespace ler

namespace {
struct ResultOpLowering : public OpConversionPattern<ResultOp> {
  using OpConversionPattern<ResultOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ResultOp Op, OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {
    return success();
  }
};

struct ConvertResultToStorePass
    : public ler::impl::ConvertResultToStoreBase<ConvertResultToStorePass> {
  using ConvertResultToStoreBase::ConvertResultToStoreBase;

public:
  void runOnOperation() override {}
};
} // namespace
