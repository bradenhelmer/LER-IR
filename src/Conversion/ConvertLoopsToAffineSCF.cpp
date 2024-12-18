// ConvertLoopsToAffineSCF.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ConvertLoopsToAffineSCF pass implementation.
#include <ler-ir/Analysis/Misc.h>
#include <ler-ir/Passes.h>
#include <map>

using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::Attribute;
using mlir::Block;
using mlir::BlockArgument;
using mlir::IntegerAttr;
using mlir::Region;
using mlir::StringAttr;
using mlir::SymbolRefAttr;
using mlir::affine::AffineApplyOp;
using mlir::affine::AffineDialect;
using mlir::affine::AffineForOp;
using mlir::affine::AffineMinOp;
using mlir::arith::ConstantIndexOp;
using mlir::func::ReturnOp;
using mlir::memref::StoreOp;
using mlir::scf::ConditionOp;
using mlir::scf::SCFDialect;
using mlir::scf::WhileOp;
using mlir::scf::YieldOp;
using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTLOOPSTOAFFINESCF
#include <ler-ir/Passes.h.inc>
} // namespace ler

namespace {
static std::map<std::string, BlockArgument> NewAffineBlkArgs;

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
    BlockArgument BlkArg = Op->getRegion(0).getArgument(0);
    int64_t LB = Op->template getAttrOfType<IntegerAttr>("LowerBound").getInt();

    // First resolve upper bound of for loop.
    Attribute UBAttr = Op->getAttr("UpperBound");

    if (auto SymRef = dyn_cast<SymbolRefAttr>(UBAttr)) {

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
      if ((BlkArg =
               NewAffineBlkArgs[SymRef.getLeafReference().getValue().str()])) {

        auto ApplyMap =
            AffineMap::get(1, 0, {ReWriter.getAffineConstantExpr(MaxUB)},
                           ReWriter.getContext());

        auto ApplyOp =
            ReWriter.create<AffineApplyOp>(UNKNOWN_LOC, ApplyMap, BlkArg);

        // Now ApplyOp.getResult() is a valid dimension operand for your UBMap
        UBMap = AffineMap::get(1, 0, ReWriter.getAffineDimExpr(0),
                               ReWriter.getContext());
        UBOperands = {ApplyOp.getResult()};

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
                         .getValue()
                         .str()] = Op->getRegion(0).getArgument(0);
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

    // Create iteration bounds.
    ReWriter.setInsertionPoint(Op);
    auto StartIterCnst = ReWriter.create<mlir::arith::ConstantOp>(
        UNKNOWN_LOC, ReWriter.getI64IntegerAttr(0));
    auto EndIterCnst = ReWriter.create<mlir::arith::ConstantOp>(
        UNKNOWN_LOC, ReWriter.getI64IntegerAttr(1000));

    // Create while loop
    auto LoweredWhile = ReWriter.create<WhileOp>(
        Op.getLoc(), mlir::TypeRange{ReWriter.getI64Type()},
        mlir::ValueRange{StartIterCnst});

    // Create before block with condition checking operations.
    Block *Before =
        ReWriter.createBlock(&LoweredWhile.getBefore(), {},
                             mlir::TypeRange{StartIterCnst.getType()});
    ReWriter.setInsertionPointToStart(Before);
    auto CounterVar = LoweredWhile.getBefore().addArgument(
        ReWriter.getI64Type(), Op.getLoc());
    auto CompareOp = ReWriter.create<mlir::arith::CmpIOp>(
        Op.getLoc(), mlir::arith::CmpIPredicate::slt, CounterVar, EndIterCnst);
    ReWriter.create<ConditionOp>(Op.getLoc(), CompareOp, CounterVar);

    // Create after block with actual while body
    ReWriter.createBlock(&LoweredWhile.getAfter(), {},
                         mlir::TypeRange{StartIterCnst.getType()});

    ReWriter.cloneRegionBefore(Op.getRegion(), &LoweredWhile.getAfter().back());
    ReWriter.eraseBlock(&LoweredWhile.getAfter().back());

    Block *After = &LoweredWhile.getAfter().front();

    ReWriter.setInsertionPointToEnd(After);

    auto CounterVarA = After->addArgument(ReWriter.getI64Type(), Op.getLoc());

    auto IncCnst = ReWriter.create<mlir::arith::ConstantOp>(
        Op.getLoc(), ReWriter.getI64IntegerAttr(1));

    auto Next =
        ReWriter.create<mlir::arith::AddIOp>(Op.getLoc(), CounterVarA, IncCnst);

    ReWriter.create<YieldOp>(Op.getLoc(), mlir::ValueRange{Next});

    ReWriter.replaceOp(Op, LoweredWhile->getResults());

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
      if (isa<ResultOp>(User) || isa<StoreOp>(User))
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

struct LoopTerminatorOpLowering : public OpConversionPattern<LoopTerminatorOp> {

  using OpConversionPattern<LoopTerminatorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoopTerminatorOp Op, OpAdaptor Adaptor,
                  ConversionPatternRewriter &ReWriter) const override {
    Op.erase();
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
        .addIllegalOp<LoopTerminatorOp, ProductionForLoopOp, RegularForLoopOp,
                      ResultOp, SummationForLoopOp, WhileLoopOp, VariableOp>();

    RewritePatternSet Patterns(&Ctx);
    Patterns.add<LoopTerminatorOpLowering, ProductionForLoopOpLowering,
                 SummationForLoopOpLowering, RegularForLoopOpLowering,
                 WhileLoopOpLowering, VariableOpLowering>(&Ctx);
    Patterns.add<ResultOpLowering>(&Ctx, &Op);

    if (failed(applyFullConversion(Op, AffineSCFTarget, std::move(Patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
