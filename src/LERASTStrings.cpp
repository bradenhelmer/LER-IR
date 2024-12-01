// LERASTStrings.cpp
// ~~~~~~~~~~~~~~~~~~
// Virtual 'getStrRep' method implementations for AST classes.
#include <ler-ir/LERFrontend.h>

using llvm::StringRef;
namespace ler {

void LERTree::print() {
  OUTS << "LER AST for source: \n"
       << LERSource.getBufferStart() << '\n' << getStrRep() << "\n";
}

StringRef LERLoopNest::getStrRep() {
  if (StrRep.empty()) {
    std::stringstream Stream;
    auto LoopCount = getLoopCount();
    for (int i = 0; i < LoopCount; ++i) {
      const auto &Loop = Loops[i];
      Stream << std::string(i * 2, ' ') << Loop->getStrRep().str() << '\n';
    }
    Stream << std::string(LoopCount * 2 + 1, ' ')
           << ExprResult->getStrRep().str();
    StrRep = Stream.str();
  }
  return StringRef(StrRep);
}
StringRef LERExpressionResultPair::getStrRep() {
  if (StrRep.empty()) {
    std::stringstream Stream;
    Stream << Expression->getStrRep().str() << " = "
           << Result->getStrRep().str();
    StrRep = Stream.str();
  }
  return StringRef(StrRep);
}

StringRef LERTree::getStrRep() {
  if (StrRep.empty()) {
    std::stringstream Stream;
    for (const auto &Stmt : Statements) {
      Stream << Stmt->getStrRep().str() << '\n';
    }
    StrRep = Stream.str();
  }
  return StringRef(StrRep);
}

StringRef LERForLoop::getStrRep() {
  if (StrRep.empty()) {
    std::stringstream Stream;
    Stream << "ForLoop: " << getTokenName(Kind) << " " << LoopIdxVar << "="
           << LBound << "->" << UBound;
    StrRep = Stream.str();
  }
  return StringRef(StrRep);
}

StringRef LERWhileLoop::getStrRep() {
  if (StrRep.empty()) {
    std::stringstream Stream;
    Stream << "WhileLoop: ";
    if (ConditionExpression) {
      Stream << "Condition (" << ConditionExpression->getStrRep().str() << ")";
    }
    StrRep = Stream.str();
  }
  return StringRef(StrRep);
}

StringRef LERArrayAccessExpression::getStrRep() {
  if (StrRep.empty()) {
    std::stringstream Stream;
    Stream << ArrVar->getStrRep().str() << '[';
    auto IndexCount = Indicies.size();
    for (int i = 0; i < IndexCount; ++i) {
      Stream << Indicies[i]->getStrRep().str();
      if (i < IndexCount - 1)
        Stream << ',';
    }
    Stream << ']';
    StrRep = Stream.str();
  }
  return StringRef(StrRep);
}

StringRef LERConstantExpression::getStrRep() {
  if (StrRep.empty())
    StrRep = std::to_string(Val);
  return StringRef(StrRep);
}

StringRef LERBinaryOpExpression::getStrRep() {
  if (StrRep.empty()) {
    std::stringstream Stream;
    Stream << LHS->getStrRep().str() << " " << getOperatorString(Operator)
           << " " << RHS->getStrRep().str();
    StrRep = Stream.str();
  }
  return StringRef(StrRep);
}

StringRef LERFunctionCallExpression::getStrRep() {
  if (StrRep.empty()) {
    std::stringstream Stream;
    Stream << FuncName->getStrRep().str() << '(';
    auto ParamCount = Parameters.size();
    for (int i = 0; i < ParamCount; ++i) {
      Stream << Parameters[i]->getStrRep().str();
      if (i < ParamCount - 1)
        Stream << ',';
    }
    Stream << ')';
    StrRep = Stream.str();
  }
  return StringRef(StrRep);
}

StringRef LERVarExpression::getStrRep() {
  if (StrRep.empty())
    StrRep = VarName;
  return StringRef(VarName);
}

StringRef LERParenExpression::getStrRep() {
  if (StrRep.empty()) {
    std::stringstream Stream;
    Stream << "(" << Expression->getStrRep().str() << ")";
    StrRep = Stream.str();
  }
  return StringRef(StrRep);
}
} // namespace ler
