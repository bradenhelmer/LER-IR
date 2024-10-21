// ConvertArrayAccToMemref.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ConvertArrayAccToMemref pass implementation.

#include <ler-ir/Passes.h>

using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTARRAYACCTOMEMREF
#include <ler-ir/Passes.h.inc>
} // namespace ler

namespace {

struct ArrayAccLowering : public OpConversionPattern<ArrayAccessOp> {
  using OpConversionPattern<ArrayAccessOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayAccessOp Op, OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {
    llvm::outs() << Op->getParentOp()->getName() << '\n';
    return success();
  }
};

struct ConvertArrayAccToMemrefPass
    : public ler::impl::ConvertArrayAccToMemrefBase<
          ConvertArrayAccToMemrefPass> {
  using ConvertArrayAccToMemrefBase::ConvertArrayAccToMemrefBase;

public:
  void runOnOperation() override {
    auto &Ctx = getContext();
    ConversionTarget MemRefTarget(Ctx);

    MemRefTarget.addLegalDialect<ArithDialect, BuiltinDialect, FuncDialect,
                                 LERDialect, MemRefDialect>();
    MemRefTarget.addIllegalOp<ArrayAccessOp>();

    RewritePatternSet Patterns(&Ctx);

    auto Op = getOperation();
    if (failed(applyFullConversion(Op, MemRefTarget, std::move(Patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
