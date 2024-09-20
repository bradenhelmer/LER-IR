// LERASTPrinters.cpp
// ~~~~~~~~~~~~~~~~~~
// Virtual 'print' method implementation for AST classes.
// Helps for debugging the Parser.
#include <ler-ir/LERFrontend.h>

namespace ler {

void LERStatement::print() {
  OUTS << "LER AST for source:\n-------------------\n"
       << LERSource.getBufferStart() << "\n";
  auto LoopCount = getLoopCount();
  for (int i = 0; i < LoopCount; ++i) {
    const auto &Loop = Loops[i];
    Loop->print(i);
  }
  Expression->print(LoopCount);
  OUTS << " = ";
  Result->print();
  OUTS << '\n';
}

void LERForLoop::print(uint8_t Indent) {
  PRINT_INDENT();
  OUTS << "ForLoop: " << getTokenName(Kind) << " " << LoopIdxVar << "="
       << LBound << "->" << UBound << "\n";
}

void LERWhileLoop::print(uint8_t Indent) {
  PRINT_INDENT();
  OUTS << "WhileLoop: ";
  if (ConditionExpression) {
    OUTS << "Condition (";
    ConditionExpression->print();
    OUTS << ")";
  }
  OUTS << '\n';
}

void LERArrayAccessExpression::print(uint8_t Indent) {
  ArrVar->print(Indent);
  OUTS << '[';
  auto IndexCount = Indicies.size();
  for (int i = 0; i < IndexCount; ++i) {
    Indicies[i]->print();
    if (i < IndexCount - 1)
      OUTS << ',';
  }
  OUTS << ']';
}

void LERConstantExpression::print(uint8_t Indent) {
  PRINT_INDENT();
  OUTS << Value;
}

void LERBinaryOpExpression::print(uint8_t Indent) {
  LHS->print(Indent);
  OUTS << " " << getOperatorString(Operator) << " ";
  RHS->print();
}

void LERFunctionCallExpression::print(uint8_t Indent) {
  FuncName->print();
  OUTS << '(';
  auto ParamCount = Parameters.size();
  for (int i = 0; i < ParamCount; ++i) {
    Parameters[i]->print();
    if (i < ParamCount - 1)
      OUTS << ',';
  }

  OUTS << ')';
}

void LERVarExpression::print(uint8_t Indent) {
  PRINT_INDENT();
  OUTS << VarName;
}

void LERParenExpression::print(uint8_t Indent) {
  PRINT_INDENT()
  OUTS << "(";
  Expression->print();
  OUTS << ")";
}

} // namespace ler
