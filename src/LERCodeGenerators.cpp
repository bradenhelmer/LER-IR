// LERCodeGenerators.cpp
// ~~~~~~~~~~~~~~~~~~~~~
// Virtual method 'codeGen' implementations for lowering the LER AST into the
// LER MLIR dialect.
#include <ler-ir/Analysis/Misc.h>
#include <ler-ir/IR/LERDialect.h>
#include <ler-ir/LERFrontend.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>

using namespace ler;
using mlir::Block;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::SymbolRefAttr;
using mlir::ValueRange;

using mlir::func::FuncDialect;
using mlir::func::FuncOp;

extern llvm::cl::opt<std::string> InputFilename;

namespace {
static MLIRContext Context;
static OpBuilder Builder(&Context);
} // namespace

#define UNKNOWN_LOC Builder.getUnknownLoc()

ModuleOp LERStatement::codeGen() {
  Context.loadDialect<LERDialect, FuncDialect>();
  auto LERModule = Builder.create<ModuleOp>(UNKNOWN_LOC, InputFilename);
  LERModule->setAttr("ler.Source",
                     Builder.getStringAttr(LERSource.getBuffer()));
  Builder.setInsertionPointToStart(LERModule.getBody());

  auto MainFunc =
      FuncOp::create(UNKNOWN_LOC, "main", Builder.getFunctionType({}, {}));
  LERModule.push_back(MainFunc);
  Builder.setInsertionPointToStart(MainFunc.addEntryBlock());

  for (const auto &Loop : Loops) {
    Loop->codeGen();
  }

  auto E = Expression->codeGen();
  Builder.create<ResultOp>(UNKNOWN_LOC, E, Result->codeGen());

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
  auto &FLBlock = ForLoop->getRegion(0).emplaceBlock();

  auto BlkArg = FLBlock.addArgument(Builder.getIndexType(), ForLoop->getLoc());
  insertIdxBlkArgMap(LoopIdxVar, BlkArg);

  Builder.setInsertionPointToStart(&FLBlock);
}

Value LERVarExpression::codeGen() {
  auto V = Builder.create<VariableOp>(UNKNOWN_LOC, VarName, Subscripts);
  return V;
}

Value LERArrayAccessExpression::codeGen() {
  SmallVector<Value, 8> IndexVals;
  for (const auto &Index : Indicies)
    IndexVals.push_back(Index->codeGen());
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
