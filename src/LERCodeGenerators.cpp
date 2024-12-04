// LERCodeGenerators.cpp
// ~~~~~~~~~~~~~~~~~~~~~
// Virtual method 'codeGen' implementations for lowering the LER AST into the
// LER MLIR dialect.
#include <ler-ir/Analysis/Misc.h>
#include <ler-ir/IR/LERDialect.h>
#include <ler-ir/LERFrontend.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>

using namespace ler;
using mlir::Block;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::SymbolRefAttr;
using mlir::ValueRange;

using mlir::arith::ArithDialect;
using mlir::arith::IndexCastOp;
using mlir::func::FuncDialect;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;

extern llvm::cl::opt<std::string> InputFilename;

namespace {
static MLIRContext Context;
static OpBuilder Builder(&Context);
FuncOp MainFunc;
static bool InLoopNest = false;
} // namespace

#define UNKNOWN_LOC Builder.getUnknownLoc()

ModuleOp LERTree::codeGen() {
  Context.loadDialect<LERDialect, FuncDialect, ArithDialect>();
  auto LERModule = Builder.create<ModuleOp>(UNKNOWN_LOC, InputFilename);
  LERModule->setAttr("ler.Source",
                     Builder.getStringAttr(LERSource.getBuffer()));
  Builder.setInsertionPointToStart(LERModule.getBody());

  MainFunc =
      FuncOp::create(UNKNOWN_LOC, "main", Builder.getFunctionType({}, {}));
  LERModule.push_back(MainFunc);
  Builder.setInsertionPointToStart(MainFunc.addEntryBlock());

  for (const auto &Stmt : Statements) {
    Stmt->codeGen();
  }

  Builder.create<ReturnOp>(Builder.getUnknownLoc());

  return LERModule;
}

void LERLoopNest::codeGen() {

  InLoopNest = true;
  for (const auto &Loop : Loops)
    Loop->codeGen();
  ExprResult->codeGen();
  Builder.setInsertionPointToEnd(&MainFunc.getBody().back());
  InLoopNest = false;
}

void LERExpressionResultPair::codeGen() {
  auto E = Expression->codeGen();
  Builder.create<ResultOp>(UNKNOWN_LOC, E, Result->codeGen());
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
  auto &FLBlock = ForLoop->getRegion(0).emplaceBlock();

  auto BlkArg = FLBlock.addArgument(Builder.getIndexType(), ForLoop->getLoc());
  insertIdxBlkArgMap(LoopIdxVar, BlkArg);

  Builder.setInsertionPointToEnd(&FLBlock);
  Builder.create<LoopTerminatorOp>(UNKNOWN_LOC);

  Builder.setInsertionPointToStart(&FLBlock);
}

Value LERVarExpression::codeGen() {
  auto V = Builder.create<VariableOp>(UNKNOWN_LOC, VarName, Subscripts);
  return V;
}

Value LERArrayAccessExpression::codeGen() {
  SmallVector<Value, 8> IndexVals;
  for (const auto &Index : Indicies) {
    auto IndexVal = Index->codeGen();

    if (isArithOp(IndexVal.getDefiningOp())) {
      IndexVal = Builder.create<IndexCastOp>(IndexVal.getLoc(),
                                             Builder.getIndexType(), IndexVal);
    }

    IndexVals.push_back(IndexVal);
  }
  return Builder.create<ArrayAccessOp>(
      UNKNOWN_LOC, Builder.getI64Type(),
      SymbolRefAttr::get(Builder.getStringAttr(ArrVar->getStrRep())),
      IndexVals);
}

Value LERBinaryOpExpression::codeGen() {
  auto L = LHS->codeGen();
  auto R = RHS->codeGen();
  Value BinOP;
  switch (Operator) {
  case ADD:
    BinOP = Builder.create<AddOp>(UNKNOWN_LOC, L, R);
    break;
  case SUB:
    BinOP = Builder.create<SubOp>(UNKNOWN_LOC, L, R);
    break;
  case MUL:
    BinOP = Builder.create<MulOp>(UNKNOWN_LOC, L, R);
    break;
  case DIV:
    BinOP = Builder.create<DivOp>(UNKNOWN_LOC, L, R);
    break;
  default:
    break;
  }

  return BinOP;
}

Value LERConstantExpression::codeGen() {
  auto C = Builder.create<ConstantOp>(UNKNOWN_LOC, Val);
  return C;
}

Value LERFunctionCallExpression::codeGen() {
  SmallVector<Value, 16> ParameterValues;
  for (const auto &Param : Parameters)
    ParameterValues.push_back(Param->codeGen());
  auto FC = Builder.create<FunctionCallOp>(
      UNKNOWN_LOC, Builder.getI64Type(),
      SymbolRefAttr::get(Builder.getStringAttr(FuncName->getStrRep())),
      ParameterValues);
  return FC;
}

Value LERParenExpression::codeGen() {
  auto P = Builder.create<ParenExprOp>(UNKNOWN_LOC, Builder.getI64Type(),
                                       Expression->codeGen());
  return P;
}
