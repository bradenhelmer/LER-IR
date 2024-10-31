// ConvertLoopsToAffineSCF.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ConvertLoopsToAffineSCF pass implementation.
#include <ler-ir/Analysis/Misc.h>
#include <ler-ir/Passes.h>

using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::Attribute;
using mlir::Block;
using mlir::BlockArgument;
using mlir::IntegerAttr;
using mlir::Region;
using mlir::StringAttr;
using mlir::SymbolRefAttr;
using mlir::affine::AffineDialect;
using mlir::affine::AffineForOp;
using mlir::affine::AffineMinOp;
using mlir::arith::ConstantIndexOp;
using mlir::func::ReturnOp;
using mlir::memref::StoreOp;
using mlir::scf::SCFDialect;
using mlir::scf::WhileOp;
using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTLOOPSTOAFFINESCF
#include <ler-ir/Passes.h.inc>
} // namespace ler

namespace {
StringMap<BlockArgument> NewAffineBlkArgs;

BlockArgument getNewAffineBlkArgFromName(StringRef Var) {
  if (NewAffineBlkArgs.find(Var) != NewAffineBlkArgs.end()) {
    return NewAffineBlkArgs[Var];
  }
  return nullptr;
}

static void flattenRegion(Region &Region) {
  if (Region.empty() || Region.hasOneBlock()) {
    return;
  }

  auto *NewBlock = new Block();

  for (auto &Blk : llvm::make_early_inc_range(Region.getBlocks())) {
    NewBlock->getOperations().splice(NewBlock->end(), Blk.getOperations());
    if (&Blk != NewBlock)
      Blk.erase();
  }

  if (Region.empty())
    Region.push_back(NewBlock);
}

template <typename LERForLoopOp, typename AffineOp = AffineForOp>
struct ForLoopLowering : public OpConversionPattern<LERForLoopOp> {
  using OpConversionPattern<LERForLoopOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(LERForLoopOp Op,
                  typename OpConversionPattern<LERForLoopOp>::OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {

    AffineForOp ForOp;
    BlockArgument BlkArg;
    int64_t LB = Op->template getAttrOfType<IntegerAttr>("LowerBound").getInt();

    // First resolve upper bound of for loop.
    Attribute UBAttr = Op->getAttr("UpperBound");

    if (auto SymRef = dyn_cast<SymbolRefAttr>(UBAttr)) {

      BlkArg = Op->getRegion(0).getArgument(0);
      auto Uses = BlkArg.getUses();

      unsigned long MaxUB = 1000000;

      for (auto &Use : Uses) {
        if (LoadOp Load = dyn_cast<LoadOp>(Use.getOwner())) {
          auto Alloc = dyn_cast<AllocOp>(Load.getMemRef().getDefiningOp());
          auto Shape = Alloc.getType().getShape();
          auto Size = Shape[Use.getOperandNumber() - 1];
          if (Size < MaxUB)
            MaxUB = Size;
        }
      }

      // Get lower bound map
      auto LBMap = AffineMap::get(0, 0, ReWriter.getAffineConstantExpr(LB));
      SmallVector<Value, 4> LBOperands;

      // Upperbound maps
      AffineMap UBMap;
      SmallVector<Value, 4> UBOperands;

      // Are we referencing another block arg here? If so, we need to create a
      // AffineMinOp operation to ensure we dont go out of bounds.
      if ((BlkArg = getNewAffineBlkArgFromName(
               SymRef.getLeafReference().getValue()))) {

        auto MinMap = AffineMap::get(1, 0,
                                     {ReWriter.getAffineDimExpr(0),
                                      ReWriter.getAffineConstantExpr(MaxUB)},
                                     ReWriter.getContext());
        auto MinOp = ReWriter.create<AffineMinOp>(UNKNOWN_LOC, MinMap, BlkArg);
        UBMap = AffineMap::get(0, 1, ReWriter.getAffineSymbolExpr(0));
        UBOperands = {MinOp->getResult(0)};

      } else {
        UBMap = AffineMap::get(0, 0, ReWriter.getAffineConstantExpr(MaxUB));
      }

      ForOp = ReWriter.create<AffineForOp>(UNKNOWN_LOC, LBOperands, LBMap,
                                           UBOperands, UBMap, 1);
    } else {
      ForOp = ReWriter.create<AffineForOp>(
          UNKNOWN_LOC, dyn_cast<IntegerAttr>(UBAttr).getInt(), LB, 1);
    }

    NewAffineBlkArgs[Op->template getAttrOfType<StringAttr>("LoopIdxVar")
                         .getValue()] = BlkArg;

    ReWriter.inlineRegionBefore(Op->getRegion(0), ForOp.getRegion(),
                                ForOp.getRegion().end());

    auto &BB0 = ForOp.getRegion().getBlocks().front();
    auto &BB1 = ForOp.getRegion().getBlocks().back();
    BB1.getOperations().splice(BB1.end(), BB0.getOperations());
    BB0.erase();

    Op->erase();

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

struct ResultOpLowering : public OpConversionPattern<ResultOp> {
private:
  ModuleOp *Module;

public:
  ResultOpLowering(MLIRContext *Ctx, ModuleOp *Module)
      : OpConversionPattern<ResultOp>(Ctx), Module(Module) {};

  LogicalResult
  matchAndRewrite(ResultOp Op, OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {
    if (auto Load = dyn_cast<LoadOp>(Op.getLocation().getDefiningOp())) {
      auto Ref = Load.getMemRef();
      auto Indicies = Load.getIndices();

      auto Store = ReWriter.create<StoreOp>(UNKNOWN_LOC, Op.getExpression(),
                                            Ref, Indicies);
      Load.erase();
    }

    else if (auto Var =
                 dyn_cast<VariableOp>(Op.getLocation().getDefiningOp())) {
      ReWriter.setInsertionPointToStart(
          &Module->lookupSymbol("main")->getRegion(0).front());
      auto VarAlloc = ReWriter.create<AllocOp>(
          UNKNOWN_LOC, mlir::MemRefType::get({1}, ReWriter.getI64Type()));

      ReWriter.setInsertionPoint(Op);
      auto Zeroth = ReWriter.create<ConstantIndexOp>(UNKNOWN_LOC, 0);
      auto Store = ReWriter.create<StoreOp>(UNKNOWN_LOC, Op.getExpression(),
                                            VarAlloc, Zeroth.getResult());
      Var.erase();
    }
    Op.erase();
    return success();
  }
};

struct VariableOpLowering : public OpConversionPattern<VariableOp> {
  using OpConversionPattern<VariableOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VariableOp Op, OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {
    bool UsedInResult = false;
    auto Uses = Op->getUses();
    for (const auto &Use : Uses) {
      auto User = Use.getOwner();
      if (isa<ResultOp>(User))
        UsedInResult = true;
    }

    if (!UsedInResult) {
      auto Constant = ReWriter.create<mlir::arith::ConstantOp>(
          Op.getLoc(), ReWriter.getI64IntegerAttr(100));
      ReWriter.replaceOp(Op, Constant.getResult());
    }
    return success();
  }
};

struct ConvertLoopsToAffineSCFPass
    : public ler::impl::ConvertLoopsToAffineSCFBase<
          ConvertLoopsToAffineSCFPass> {
  using ConvertLoopsToAffineSCFBase::ConvertLoopsToAffineSCFBase;

  void runOnOperation() override {
    auto &Ctx = getContext();
    auto Op = getOperation();

    ConversionTarget AffineSCFTarget(Ctx);
    AffineSCFTarget
        .addLegalDialect<AffineDialect, ArithDialect, BuiltinDialect,
                         FuncDialect, LERDialect, MemRefDialect, SCFDialect>();

    AffineSCFTarget
        .addIllegalOp<ProductionForLoopOp, RegularForLoopOp, ResultOp,
                      SummationForLoopOp, WhileLoopOp, VariableOp>();

    RewritePatternSet Patterns(&Ctx);
    Patterns
        .add<ProductionForLoopOpLowering, SummationForLoopOpLowering,
             RegularForLoopOpLowering, WhileLoopOpLowering, VariableOpLowering>(
            &Ctx);
    Patterns.add<ResultOpLowering>(&Ctx, &Op);

    OpBuilder Builder(&Ctx);
    Builder.setInsertionPointToEnd(
        &Op.lookupSymbol("main")->getRegion(0).front());
    Builder.create<ReturnOp>(Builder.getUnknownLoc());

    if (failed(applyFullConversion(Op, AffineSCFTarget, std::move(Patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
