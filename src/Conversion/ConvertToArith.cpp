// ConvertToArith.cpp
// ~~~~~~~~~~~~~~~~~~
// ConvertToArith pass implementation.
#include <ler-ir/LERUtils.h>
#include <ler-ir/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>

using mlir::BuiltinDialect;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::RewritePatternSet;
using mlir::success;
using mlir::arith::ArithDialect;
using mlir::func::FuncDialect;
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
    auto NewBinOp = ReWriter.create<ArithBinOp>(Op.getLoc(), Adaptor.getLHS(),
                                                Adaptor.getRHS());
    ReWriter.replaceAllUsesWith(Op, NewBinOp);
    ReWriter.eraseOp(Op);
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
    ReWriter.replaceAllUsesWith(Op, NewConstantOp);
    ReWriter.eraseOp(Op);
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
    ArithTarget.addLegalDialect<LERDialect, ArithDialect, BuiltinDialect,
                                FuncDialect>();
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
