// ConvertToLLVM.cpp
// ~~~~~~~~~~~~~~~~~
// Converts all intermediary dialects down to LLVM.
#include <ler-ir/Passes.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>

using mlir::LLVMConversionTarget;
using mlir::LLVMTypeConverter;
using namespace ler;

namespace ler {
#define GEN_PASS_DEF_CONVERTTOLLVM
#include <ler-ir/Passes.h.inc>
} // namespace ler

struct ConvertToLLVMPass
    : public ler::impl::ConvertToLLVMBase<ConvertToLLVMPass> {
  using ConvertToLLVMBase::ConvertToLLVMBase;

  void runOnOperation() override {
    LLVMConversionTarget LLVMTarget(getContext());

    auto &Ctx = getContext();
    Ctx.loadDialect<ControlFlowDialect>();

    LLVMTarget.addLegalOp<ModuleOp>();

    LLVMTypeConverter TC(&getContext());

    RewritePatternSet patterns(&getContext());
    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(TC, patterns);
    mlir::populateFinalizeMemRefToLLVMConversionPatterns(TC, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(TC, patterns);
    mlir::populateFuncToLLVMConversionPatterns(TC, patterns);

    if (failed(applyFullConversion(getOperation(), LLVMTarget,
                                   std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
