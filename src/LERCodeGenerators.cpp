// LERCodeGenerators.cpp
// ~~~~~~~~~~~~~~~~~~~~~
// Virtual method 'codeGen' implementations for lowering the LER AST into the
// LER MLIR dialect.
#include <ler-ir/IR/LERDialect.h>
#include <ler-ir/LERFrontend.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/IR/Builders.h>

using namespace ler;
using mlir::Block;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;

extern llvm::cl::opt<std::string> InputFilename;

namespace {
static MLIRContext Context;
static OpBuilder Builder(&Context);
} // namespace

#define UNKNOWN_LOC Builder.getUnknownLoc()

ModuleOp LERStatement::codeGen() {
  Context.loadDialect<LERDialect>();
  auto LERModule = Builder.create<ModuleOp>(UNKNOWN_LOC, InputFilename);
  LERModule->setAttr("Source", Builder.getStringAttr(LERSource.getBuffer()));
  Builder.setInsertionPointToStart(LERModule.getBody());

  for (const auto &Loop : Loops) {
    Loop->codeGen();
  }

  return LERModule;
}

void LERWhileLoop::codeGen() {
  auto WhileLoop = Builder.create<WhileLoopOp>(
      UNKNOWN_LOC, ConditionExpression->getStrRep(), Subscripts);
  auto *WBlock = new Block();
  WhileLoop->getRegion(0).push_back(WBlock);
  Builder.setInsertionPointToStart(WBlock);
}

void LERForLoop::codeGen() {
  Operation *ForLoop;

  switch (Kind) {
  case PRODUCT: {
    ForLoop = Builder.create<ProductionForLoopOp>(UNKNOWN_LOC, LBound, UBound,
                                                  LoopIdxVar);
    break;
  }
  case REGULAR_FOR: {
    ForLoop = Builder.create<RegularForLoopOp>(UNKNOWN_LOC, LBound, UBound,
                                               LoopIdxVar);
    break;
  }
  case SUMMATION: {
    ForLoop = Builder.create<SummationForLoopOp>(UNKNOWN_LOC, LBound, UBound,
                                                 LoopIdxVar);
    break;
  }
  default:
    return;
  }
  auto *FLBlock = new Block();
  ForLoop->getRegion(0).push_back(FLBlock);
  Builder.setInsertionPointToStart(FLBlock);
}

Value LERVarExpression::codeGen() { return nullptr; }
Value LERArrayAccessExpression::codeGen() { return nullptr; }
Value LERBinaryOpExpression::codeGen() { return nullptr; }
Value LERConstantExpression::codeGen() { return nullptr; }
Value LERFunctionCallExpression::codeGen() { return nullptr; }
Value LERParenExpression::codeGen() { return nullptr; }
