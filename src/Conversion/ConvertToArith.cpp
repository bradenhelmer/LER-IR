// ConvertToArith.cpp
// ~~~~~~~~~~~~~~~~~~
// ConvertToArith pass implementation.
#include <ler-ir/LERUtils.h>
#include <ler-ir/Passes.h>
#include <mlir/IR/BuiltinDialect.h>

using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTTOARITH
#include <ler-ir/Passes.h.inc>
} // namespace ler

namespace {

template <typename LERBinOp, typename ArithBinOp>
struct BinaryOpLowering : public OpConversionPattern<LERBinOp> {
  using OpConversionPattern<LERBinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LERBinOp Op,
                  typename OpConversionPattern<LERBinOp>::OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {
    auto NewBinOp = ReWriter.create<ArithBinOp>(
        Op.getLoc(), Op.getType(), Adaptor.getLHS(), Adaptor.getRHS());
    ReWriter.replaceOp(Op, NewBinOp.getResult());
    return success();
  }
};

using AddOpLowering = BinaryOpLowering<AddOp, mlir::arith::AddIOp>;
using SubOpLowering = BinaryOpLowering<SubOp, mlir::arith::SubIOp>;
using MulOpLowering = BinaryOpLowering<MulOp, mlir::arith::MulIOp>;
using DivOpLowering = BinaryOpLowering<DivOp, mlir::arith::DivSIOp>;

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp Op, OpAdaptor As,
                  ConversionPatternRewriter &ReWriter) const override {
    auto NewConstantOp = ReWriter.create<mlir::arith::ConstantOp>(
        Op.getLoc(), ReWriter.getI64IntegerAttr(Op.getValueAttr().getInt()));
    ReWriter.replaceOp(Op, NewConstantOp.getResult());
    return success();
  }
};

struct ConvertToArithPass
    : public ler::impl::ConvertToArithBase<ConvertToArithPass> {
  using ConvertToArithBase::ConvertToArithBase;

private:
  void populateArithConversionPatterns(RewritePatternSet &Patterns) {
    Patterns.add<AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering,
                 ConstantOpLowering>(&getContext());
  }

public:
  void runOnOperation() override {
    auto &Ctx = getContext();
    ConversionTarget ArithTarget(Ctx);
    ArithTarget.addLegalDialect<ArithDialect, BuiltinDialect, FuncDialect,
                                LERDialect>();
    ArithTarget.addIllegalOp<AddOp, SubOp, MulOp, DivOp, ConstantOp>();

    RewritePatternSet Patterns(&Ctx);
    populateArithConversionPatterns(Patterns);

    auto Op = getOperation();
    if (failed(applyFullConversion(Op, ArithTarget, std::move(Patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
