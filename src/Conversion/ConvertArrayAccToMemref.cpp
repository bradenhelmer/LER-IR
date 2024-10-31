// ConvertArrayAccToMemref.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ConvertArrayAccToMemref pass implementation.

#include <ler-ir/Passes.h>

using mlir::MLIRContext;
using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTARRAYACCTOMEMREF
#include <ler-ir/Passes.h.inc>
} // namespace ler

namespace {

struct ArrayAccLowering : public OpConversionPattern<ArrayAccessOp> {

private:
  ModuleOp *Module;
  mutable StringMap<AllocOp> ArrayAllocMap;

  SmallVector<int64_t> getArrShape(size_t IndexCount) const {

    // Static collection of array shapes supporting upto 6 dimensions, all
    // totalling up to one million I64 elements.
    static const SmallVector<int64_t> ArrSizes[] = {
        {1000000},          {1000, 1000},          {100, 100, 100},
        {10, 10, 100, 100}, {10, 10, 10, 10, 100}, {10, 10, 10, 10, 10, 10}};

    return ArrSizes[IndexCount - 1];
  }

  void createAllocFromArrayAccOp(ArrayAccessOp *Op, OpAdaptor &Adaptor,
                                 ConversionPatternRewriter &ReWriter,
                                 StringRef ArrName) const {
    auto IdxCnt = Op->getIndicies().size();
    SmallVector<int64_t> ArrShape = getArrShape(IdxCnt);

    auto CurrBlock = ReWriter.getBlock();
    auto CurrInsertPoint = ReWriter.getInsertionPoint();

    ReWriter.setInsertionPointToStart(
        &Module->lookupSymbol("main")->getRegion(0).front());
    auto ArrAlloc = ReWriter.create<AllocOp>(
        UNKNOWN_LOC, mlir::MemRefType::get(ArrShape, ReWriter.getI64Type()));

    ReWriter.setInsertionPoint(CurrBlock, CurrInsertPoint);
    ArrayAllocMap.insert_or_assign(ArrName, ArrAlloc);
  }

public:
  ArrayAccLowering(MLIRContext *Ctx, ModuleOp *Module)
      : OpConversionPattern<ArrayAccessOp>(Ctx), Module(Module) {};

  LogicalResult
  matchAndRewrite(ArrayAccessOp Op, OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {

    auto ArrName = Op.getArrayNameAttr().getLeafReference().getValue();

    if (!ArrayAllocMap.lookup(ArrName))
      createAllocFromArrayAccOp(&Op, Adaptor, ReWriter, ArrName);

    // Create memref ops.
    auto Alloc = ArrayAllocMap.lookup(ArrName);

    auto Load = ReWriter.create<LoadOp>(UNKNOWN_LOC, Alloc, Op.getIndicies());

    ReWriter.replaceOp(Op, Load.getResult());

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
    auto Op = getOperation();
    ConversionTarget MemRefTarget(Ctx);

    MemRefTarget.addLegalDialect<ArithDialect, BuiltinDialect, FuncDialect,
                                 LERDialect, MemRefDialect>();

    MemRefTarget.addIllegalOp<ArrayAccessOp>();

    RewritePatternSet Patterns(&Ctx);
    Patterns.add<ArrayAccLowering>(&Ctx, &Op);

    if (failed(applyFullConversion(Op, MemRefTarget, std::move(Patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
