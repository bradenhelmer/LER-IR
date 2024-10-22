// ConvertLoopsToAffineSCF.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ConvertLoopsToAffineSCF pass implementation.
#include <ler-ir/Passes.h>

using mlir::affine::AffineDialect;
using mlir::affine::AffineForOp;
using mlir::scf::SCFDialect;
using mlir::scf::WhileOp;
using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTLOOPSTOAFFINESCF
#include <ler-ir/Passes.h.inc>
} // namespace ler

namespace {

template <typename LERForLoopOp, typename AffineOp = AffineForOp>
struct ForLoopLowering : public OpConversionPattern<LERForLoopOp> {
  using OpConversionPattern<LERForLoopOp>::OpConversionPattern;
private:
	 

public:
  LogicalResult
  matchAndRewrite(LERForLoopOp Op,
                  typename OpConversionPattern<LERForLoopOp>::OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {
    return success();
  }
};

using ProductionForLoopOpLowering = ForLoopLowering<ProductionForLoopOp>;
using SummationForLoopOpLowering = ForLoopLowering<SummationForLoopOp>;
using RegularForLoopOpLowering = ForLoopLowering<RegularForLoopOp>;

struct WhileLoopOpLowering : public OpConversionPattern<WhileLoopOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(WhileLoopOp Op, OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {
    return success();
  }
};

struct ConvertLoopsToAffineSCFPass
    : public ler::impl::ConvertLoopsToAffineSCFBase<
          ConvertLoopsToAffineSCFPass> {
  using ConvertLoopsToAffineSCFBase::ConvertLoopsToAffineSCFBase;

  void runOnOperation() override {
    auto &Ctx = getContext();
    ConversionTarget AffineSCFTarget(Ctx);
    AffineSCFTarget
        .addLegalDialect<AffineDialect, ArithDialect, BuiltinDialect,
                         FuncDialect, LERDialect, MemRefDialect, SCFDialect>();

    AffineSCFTarget.addIllegalOp<ProductionForLoopOp, RegularForLoopOp,
                                 SummationForLoopOp, WhileLoopOp>();

    RewritePatternSet Patterns(&Ctx);
    Patterns.add<ProductionForLoopOpLowering, SummationForLoopOpLowering,
                 RegularForLoopOpLowering, WhileLoopOpLowering>(&Ctx);

    auto Op = getOperation();
    if (failed(applyFullConversion(Op, AffineSCFTarget, std::move(Patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
