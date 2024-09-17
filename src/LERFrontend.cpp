// Frontend.cpp
// ~~~~~~~~~~~~
// Token, Lexer, and Parser implementations.
#include <ler-ir/LERCommonUtils.h>
#include <ler-ir/LERFrontend.h>
#include <string>

namespace ler {

LERLexer::LERLexer(const std::string &InputFilename) {

  auto InputBufferOrErr = llvm::MemoryBuffer::getFile(InputFilename, true);

  if (!InputBufferOrErr) {
    ERRS << "There was an error reading '" << InputFilename
         << "'!\nError: " << InputBufferOrErr.getError().message();
    std::exit(1);
  }

  LERInputBuffer = std::move(*InputBufferOrErr);
  LERBufStart = LERInputBuffer->getBufferStart();
  LERBufCurr = LERBufStart;
}

bool LERLexer::lexToken(LERToken *Out) {

  while (isWhiteSpace(*LERBufCurr)) {
    LERBufCurr++;
  }

  Out->Start = LERBufCurr;
  Out->End = LERBufCurr;

  const char Peek = *(LERBufCurr + 1);

  switch (*LERBufCurr) {
  case 0:
    Out->Kind = LER_EOF;
    return true;
  case '\n':
    Out->Kind = NEWLINE;
    break;
  case '+':
    Out->Kind = ADD;
    break;
  case '-':
    Out->Kind = SUB;
    break;
  case '*':
    Out->Kind = MUL;
    break;
  case '/':
    Out->Kind = DIV;
    break;
  case '$':
    Out->Kind = SUBSCRIPT;
    break;
  case ',':
    Out->Kind = COMMA;
    break;
  case '[':
    Out->Kind = OPEN_BRACKET;
    break;
  case ']':
    Out->Kind = CLOSE_BRACKET;
    break;
  case '(':
    Out->Kind = OPEN_PAREN;
    break;
  case ')':
    Out->Kind = CLOSE_PAREN;
    break;
  case '=':
    if (Peek == '=') {
      Out->End = ++LERBufCurr;
      Out->Kind = EQ;
      break;
    }
    Out->Kind = ASSIGN;
    break;
  case '>':
    if (Peek == '=') {
      Out->End = ++LERBufCurr;
      Out->Kind = GE;
      break;
    }
    Out->Kind = GT;
    break;
  case '<':
    if (Peek == '=') {
      Out->End = ++LERBufCurr;
      Out->Kind = LE;
      break;
    }
    Out->Kind = LT;
    break;
  case '&':
    if (Peek == '&') {
      Out->End = ++LERBufCurr;
      Out->Kind = LAND;
      break;
    }
    return false;
  case '|':
    if (Peek == '|') {
      Out->End = ++LERBufCurr;
      Out->Kind = LOR;
      break;
    }
    Out->Kind = INDEX;
    break;
  case '!':
    if (Peek == '=') {
      Out->End = ++LERBufCurr;
      Out->Kind = NE;
      break;
    }
    return false;
  case '^':
    return lexLoopIdentifier(Out);
    // clang-format off
  case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
  case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
  case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
  case 'V': case 'W': case 'X': case 'Y': case 'Z':
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z':
  case '_':
    Out->Kind = ID;
    return lexIdentifier(Out);
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
	Out->Kind = NUMBER;
    // clang-format on
    return lexNumber(Out);
  }

  LERBufCurr++;
  return true;
}

bool LERLexer::lexIdentifier(LERToken *Out) {
  do {
    LERBufCurr++;
  } while (isIdentifierChar(*LERBufCurr));

  Out->End = LERBufCurr - 1;
  return true;
}

bool LERLexer::lexLoopIdentifier(LERToken *Out) {
  LERBufCurr++;

  switch (*LERBufCurr) {
  case 'R':
    Out->Kind = REGULAR_FOR;
    break;
  case 'S':
    Out->Kind = SUMMATION;
    break;
  case 'P':
    Out->Kind = PRODUCT;
    break;
  case 'W':
    Out->Kind = WHILE;
    break;
  default:
    return false;
  }

  Out->End = LERBufCurr++;
  return true;
}

bool LERLexer::lexNumber(LERToken *Out) {
  do {
    LERBufCurr++;
  } while (isdigit(*LERBufCurr));

  Out->End = LERBufCurr - 1;
  return true;
}

void LERParser::lexAndPrintTokens() {
  auto CurrOld = Lexer->LERBufCurr;
  Lexer->LERBufCurr = Lexer->LERBufStart;

  while (Lexer->lexToken(&CurrToken) && CurrToken.Kind != LER_EOF) {
    OUTS << TokenNames[CurrToken.Kind] << '\n';
  }

  Lexer->LERBufCurr = CurrOld;
}

void LERParser::advance() {
  if (!Lexer->lexToken(&CurrToken)) {
    ERRS << "Error lexing token!\n";
    std::exit(1);
  }
}

void LERParser::parseLERStatement(LERStatement &Stmt) {
  std::unique_ptr<LERLoop> Loop;
  while ((Loop = parseLoop()))
    Stmt.addLoop(std::move(Loop));

  if (Stmt.getLoopCount() < 1) {
    ERRS << "At least one loop required in LER Statement!\n";
    std::exit(1);
  }

  Stmt.setExpression(parseExpression());
  hardMatch(ASSIGN);
  advance();
  Stmt.setResult(parseResult());
}

std::unique_ptr<LERLoop> LERParser::parseLoop() {

  if (isForLoop(CurrToken.Kind)) {
    advance();
    auto ForLoop = std::make_unique<LERForLoop>(CurrToken.Kind);
    parseForParam(ForLoop.get());
    return ForLoop;
  } else if (CurrToken.Kind == WHILE) {
    advance();
    auto WhileLoop = std::make_unique<LERWhileLoop>();
    std::string Subscript;
    while (parseSubscript(&Subscript))
      WhileLoop->addSubscript(Subscript);
    auto ConditionExpression = parseConditionExpression();
    if (ConditionExpression)
      WhileLoop->attachConditionExpression(std::move(ConditionExpression));
    hardMatch(INDEX);
    advance();
    return WhileLoop;
  } else {
    return nullptr;
  }
}

void LERParser::parseForParam(LERForLoop *ForLoop) {
  hardMatch(ID);
  ForLoop->setLoopIdxVar(CurrToken.getTokenString());
  advance();
  hardMatch(INDEX);
  advance();
  ForLoop->setLBound(parseBound());
  hardMatch(COMMA);
  advance();
  ForLoop->setUBound(parseBound());
  hardMatch(INDEX);
  advance();
}

std::string LERParser::parseBound() {
  if (softMatch(NUMBER)) {
    auto Num = CurrToken.getTokenString();
    advance();
    return Num;
  } else if (softMatch(SUB)) {
    advance();
    hardMatch(NUMBER);
    auto Num = "-" + CurrToken.getTokenString();
    advance();
    return Num;
  } else {
    hardMatch(ID);
    auto Id = CurrToken.getTokenString();
    advance();
    return Id;
  }
}

bool LERParser::parseSubscript(std::string *Out) {
  if (softMatch(SUBSCRIPT)) {
    advance();
    hardMatch(ID);
    *Out = CurrToken.getTokenString();
    advance();
    hardMatch(SUBSCRIPT);
    advance();
    return true;
  }
  return false;
}

std::unique_ptr<LERExpression> LERParser::parseConditionExpression() {
  auto LHS = parseCondition();
  if (LHS) {
    advance();
    if (softMatch(LAND) || softMatch(LOR)) {
      auto Operator = CurrToken.Kind;
      advance();
      auto RHS = parseCondition();
      return std::make_unique<LERBinaryOpExpression>(std::move(LHS),
                                                     std::move(RHS), Operator);
    }
    return LHS;
  }
  return nullptr;
}

std::unique_ptr<LERExpression> LERParser::parseCondition() {
  auto LHS = parseExpression();
  if (!isComparisonOperator(CurrToken.Kind)) {
    ERRS << "Parsing error: Expected comparison operator (==, !=, >, >=, <, "
            "<=)!\n";
    std::exit(1);
  }
  auto Operator = CurrToken.Kind;
  auto RHS = parseExpression();
  return std::make_unique<LERBinaryOpExpression>(std::move(LHS), std::move(RHS),
                                                 Operator);
}

std::unique_ptr<LERExpression> LERParser::parseExpression() {
  std::unique_ptr<LERExpression> LHS;

  switch (CurrToken.Kind) {
  case ID: {
    auto Id = CurrToken.getTokenString();
    auto VarExpr = std::make_unique<LERVarExpression>(Id);
    advance();
    if (softMatch(SUBSCRIPT)) {
      std::string Subscript;
      while (parseSubscript(&Subscript))
        VarExpr->addSubscript(Subscript);
    }
    if (softMatch(OPEN_PAREN)) {
      advance();
      auto FuncCallExpr =
          std::make_unique<LERFunctionCallExpression>(std::move(VarExpr));
      std::unique_ptr<LERExpression> Parameter;
      while ((Parameter = parseExpression()))
        FuncCallExpr->addParameter(std::move(Parameter));
      hardMatch(OPEN_BRACKET);
      advance();
      LHS = std::move(FuncCallExpr);
      break;
    }
    if (softMatch(OPEN_BRACKET)) {
      advance();
      auto ArrayAccExpr =
          std::make_unique<LERArrayAccessExpression>(std::move(VarExpr));
      std::unique_ptr<LERExpression> Index;
      while ((Index = parseExpression()))
        ArrayAccExpr->addIndex(std::move(Index));
      hardMatch(CLOSE_BRACKET);
      advance();
      LHS = std::move(ArrayAccExpr);
      break;
    }
    LHS = std::move(VarExpr);
    break;
  }
  case OPEN_PAREN: {
    advance();
    LHS = parseExpression();
    hardMatch(CLOSE_PAREN);
    advance();
    break;
  }
  case SUB: {
    advance();
    hardMatch(NUMBER);
    auto Num = "-" + CurrToken.getTokenString();
    advance();
    LHS = std::make_unique<LERConstantExpression>(
        static_cast<int64_t>(std::stol(Num)));
    break;
  }
  case NUMBER: {
    auto Num = CurrToken.getTokenString();
    advance();
    LHS = std::make_unique<LERConstantExpression>(
        static_cast<int64_t>(std::stol(Num)));
    break;
  }
  default:
    LHS = nullptr;
    break;
  }

  switch (CurrToken.Kind) {
  case COMMA:
    advance();
    break;
  case ADD:
  case SUB:
  case MUL:
  case DIV:
    return parseBinaryOpExpression(std::move(LHS), BASE);
  default:
    break;
  }

  return LHS;
}

std::unique_ptr<LERExpression>
LERParser::parseBinaryOpExpression(std::unique_ptr<LERExpression> LHS,
                                   LEROperatorPrecedence Prec) {
  LEROperatorPrecedence CurrPrec = getOperatorPrecedence(CurrToken.Kind);
  while (true) {
    if (CurrPrec < Prec)
      return std::move(LHS);

    LERTokenKind Op = CurrToken.Kind;
    advance();

    std::unique_ptr<LERExpression> RHS;
    RHS = parseExpression();

    if (!RHS)
      return nullptr;
    LEROperatorPrecedence PrevPrec = CurrPrec;
    CurrPrec = getOperatorPrecedence(CurrToken.Kind);

    if (CurrPrec < PrevPrec) {
      RHS = parseBinaryOpExpression(std::move(RHS), PrevPrec);
      if (!RHS)
        return nullptr;
    }
    LHS = std::make_unique<LERBinaryOpExpression>(std::move(LHS),
                                                  std::move(RHS), Op);
  }
}

std::unique_ptr<LERExpression> LERParser::parseResult() {
  hardMatch(ID);
  auto Id = CurrToken.getTokenString();
  auto VarExpr = std::make_unique<LERVarExpression>(Id);
  advance();
  if (softMatch(SUBSCRIPT)) {
    std::string Subscript;
    while (parseSubscript(&Subscript))
      VarExpr->addSubscript(Subscript);
  }
  if (softMatch(OPEN_PAREN)) {
    advance();
    auto FuncCallExpr =
        std::make_unique<LERFunctionCallExpression>(std::move(VarExpr));
    std::unique_ptr<LERExpression> Parameter;
    while ((Parameter = parseExpression()))
      FuncCallExpr->addParameter(std::move(Parameter));
    return FuncCallExpr;
  }
  if (softMatch(OPEN_BRACKET)) {
    advance();
    auto ArrayAccExpr =
        std::make_unique<LERArrayAccessExpression>(std::move(VarExpr));
    std::unique_ptr<LERExpression> Index;
    while ((Index = parseExpression()))
      ArrayAccExpr->addIndex(std::move(Index));
    return ArrayAccExpr;
  }

  return VarExpr;
}

} // namespace ler
