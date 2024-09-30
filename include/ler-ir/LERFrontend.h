// LERFrontend.h
// ~~~~~~~~~~
// Token, Lexer, and Parser definitions.
#ifndef LERIR_FRONTEND_H
#define LERIR_FRONTEND_H
#include <ler-ir/LERCommonUtils.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

using llvm::ArrayRef;
using llvm::MemoryBuffer;
using llvm::MemoryBufferRef;
using llvm::SmallVector;
using llvm::StringRef;
using mlir::ModuleOp;
using mlir::Value;
namespace ler {

// TOKENS
// ~~~~~~
enum LERTokenKind : uint8_t {
#define LER_TOKEN(X) X,
#include <ler-ir/LERTokenDefs.h>
  NUM_TOKENS
};

static const char *TokenNames[NUM_TOKENS] = {
#define LER_TOKEN(X) #X,
#include <ler-ir/LERTokenDefs.h>
};

inline static const char *getTokenName(LERTokenKind Kind) {
  return TokenNames[Kind];
}

inline static bool isForLoop(LERTokenKind Kind) {
  return Kind == REGULAR_FOR || Kind == SUMMATION || Kind == PRODUCT;
}

inline static const char *getOperatorString(LERTokenKind Kind) {
  switch (Kind) {
  case ADD:
    return "+";
  case SUB:
    return "-";
  case MUL:
    return "*";
  case DIV:
    return "/";
  case ASSIGN:
    return "=";
  case LAND:
    return "&&";
  case LOR:
    return "||";
  case EQ:
    return "==";
  case NE:
    return "!=";
  case GT:
    return ">";
  case GE:
    return ">=";
  case LT:
    return "<";
  case LE:
    return "<=";
  default:
    return "";
  }
}

inline static bool isComparisonOperator(LERTokenKind Kind) {
  return Kind >= EQ && Kind <= LE;
}

// Precedence for supported operators in LER.
enum LEROperatorPrecedence : uint8_t {
  ERROR = 0,
  BASE = 1,
  ADDITIVE = 2,
  MULTIPLICATIVE = 3
};

inline static LEROperatorPrecedence getOperatorPrecedence(LERTokenKind Kind) {
  switch (Kind) {
  case ADD:
  case SUB:
    return ADDITIVE;
  case MUL:
  case DIV:
    return MULTIPLICATIVE;
  default:
    return ERROR;
  }
}

struct LERToken {
  const char *Start;
  const char *End;
  LERTokenKind Kind;
  std::string getTokenString() { return std::string(Start, (End - Start) + 1); }
};

// LEXICAL ANALYSIS
// ~~~~~~~~~~~~~~~~
class LERLexer {

  // Only the parser will need access to the lexer's members.
  friend class LERParser;

  // LER input file.
  std::unique_ptr<MemoryBuffer> LERInputBuffer;

  // Buffer pointers
  const char *LERBufStart;
  const char *LERBufCurr;

  // Lexical methods
  bool lexToken(LERToken *Out);
  bool lexIdentifier(LERToken *Out);
  bool lexLoopIdentifier(LERToken *Out);
  bool lexNumber(LERToken *Out);

  // 'is' methods
  static bool isWhiteSpace(char c) { return c == ' ' | c == '\t' | c == '\r'; }
  static bool isIdentifierChar(char c) {
    return isdigit(c) || isalpha(c) || c == '_';
  }

public:
  LERLexer(const std::string &InputFilename);
};

// ABSTRACT SYNTAX TREES
// ~~~~~~~~~~~~~~~~~~~~~

class LERASTNode {
protected:
  std::string StrRep = "";

public:
  void print(uint8_t Indent = 0) { OUTS << getStrRep(); }
  virtual StringRef getStrRep() = 0;
};

// Base classes.
class LERLoop : public LERASTNode {
public:
  virtual ~LERLoop() = default;
  virtual void codeGen() = 0;
};

class LERExpression : public LERASTNode {
public:
  virtual ~LERExpression() = default;
  virtual Value codeGen() = 0;
};

// Loops
class LERForLoop : public LERLoop {
  LERTokenKind Kind;
  std::string LoopIdxVar;
  std::string LBound;
  std::string UBound;

public:
  LERForLoop(LERTokenKind Kind) : Kind(Kind) {}
  void setLoopIdxVar(std::string IdxVar) { LoopIdxVar = IdxVar; }
  void setLBound(std::string LB) { LBound = LB; }
  void setUBound(std::string UB) { UBound = UB; }
  StringRef getStrRep() override;
  void codeGen() override;
};

class LERWhileLoop : public LERLoop {
  SmallVector<std::string, 16> Subscripts;
  std::unique_ptr<LERExpression> ConditionExpression = nullptr;

public:
  void addSubscript(std::string SS) { Subscripts.push_back(SS); }
  void attachConditionExpression(std::unique_ptr<LERExpression> CE) {
    ConditionExpression = std::move(CE);
  }
  size_t getSubscriptCount() const { return Subscripts.size(); }
  StringRef getStrRep() override;
  ArrayRef<std::string> getSubscripts() const { return Subscripts; }
  void codeGen() override;
};

// Expressions
class LERVarExpression : public LERExpression {
  std::string VarName;
  SmallVector<std::string, 16> Subscripts;

public:
  LERVarExpression(std::string VarName) : VarName(VarName) {}
  void addSubscript(std::string SS) { Subscripts.push_back(SS); }
  size_t getSubscriptCount() const { return Subscripts.size(); }
  StringRef getStrRep() override;
  Value codeGen() override;
};

class LERArrayAccessExpression : public LERExpression {
  std::unique_ptr<LERVarExpression> ArrVar;
  SmallVector<std::unique_ptr<LERExpression>, 16> Indicies;

public:
  LERArrayAccessExpression(std::unique_ptr<LERVarExpression> ArrVar)
      : ArrVar(std::move(ArrVar)) {}
  void addIndex(std::unique_ptr<LERExpression> Index) {
    Indicies.push_back(std::move(Index));
  }
  StringRef getStrRep() override;
  Value codeGen() override;
};

class LERBinaryOpExpression : public LERExpression {
  std::unique_ptr<LERExpression> RHS;
  std::unique_ptr<LERExpression> LHS;
  LERTokenKind Operator;

public:
  LERBinaryOpExpression(std::unique_ptr<LERExpression> LHS,
                        std::unique_ptr<LERExpression> RHS,
                        LERTokenKind Operator)
      : LHS(std::move(LHS)), RHS(std::move(RHS)), Operator(Operator) {}
  StringRef getStrRep() override;
  Value codeGen() override;
};

class LERConstantExpression : public LERExpression {
  int64_t Val;

public:
  LERConstantExpression(int64_t Value) : Val(Value) {}
  int64_t getValue() const { return Val; }
  StringRef getStrRep() override;
  Value codeGen() override;
};

class LERFunctionCallExpression : public LERExpression {
  std::unique_ptr<LERVarExpression> FuncName;
  SmallVector<std::unique_ptr<LERExpression>, 16> Parameters;

public:
  LERFunctionCallExpression(std::unique_ptr<LERVarExpression> FuncName)
      : FuncName(std::move(FuncName)) {}
  void addParameter(std::unique_ptr<LERExpression> Parameter) {
    Parameters.push_back(std::move(Parameter));
  }
  StringRef getStrRep() override;
  Value codeGen() override;
};

class LERParenExpression : public LERExpression {
  std::unique_ptr<LERExpression> Expression;

public:
  LERParenExpression(std::unique_ptr<LERExpression> Expression)
      : Expression(std::move(Expression)) {}
  StringRef getStrRep() override;
  Value codeGen() override;
};

// Main tree
class LERStatement : public LERASTNode {
  SmallVector<std::unique_ptr<LERLoop>, 16> Loops;
  std::unique_ptr<LERExpression> Expression;
  std::unique_ptr<LERExpression> Result;

  MemoryBufferRef LERSource;

public:
  LERStatement(MemoryBufferRef LERSource) : LERSource(LERSource) {}

  void addLoop(std::unique_ptr<LERLoop> Loop) {
    Loops.push_back(std::move(Loop));
  }
  size_t getLoopCount() const { return Loops.size(); }
  void setExpression(std::unique_ptr<LERExpression> Expression) {
    this->Expression = std::move(Expression);
  }
  void setResult(std::unique_ptr<LERExpression> Result) {
    this->Result = std::move(Result);
  }
  void print();
  StringRef getStrRep() override;
  ModuleOp codeGen();
};

// PARSING
// ~~~~~~~
class LERParser {
  // The current lexed token.
  LERToken CurrToken;

  // The parser owns the lexer.
  std::unique_ptr<LERLexer> Lexer;

  // Lexes a token
  void advance();

  // Matches current token kind, errors out if not.
  void hardMatch(LERTokenKind Kind) const {
    if (CurrToken.Kind != Kind) {
      ERRS << "Parsing error! Expected " << TokenNames[Kind] << " token, got "
           << TokenNames[CurrToken.Kind] << "\n";
      std::exit(1);
    }
  }

  // Matches current token kind. Use when need to match
  // against multiple possible token kinds..
  bool softMatch(LERTokenKind Kind) const { return CurrToken.Kind == Kind; }

  // Parsing methods for each non-terminal in LER Grammar.
public:
  void parseLERStatement(LERStatement &Stmt);

private:
  std::unique_ptr<LERLoop> parseLoop();
  void parseForParam(LERForLoop *ForLoop);
  std::string parseBound();
  std::unique_ptr<LERExpression> parseConditionExpression();
  std::unique_ptr<LERExpression> parseCondition();
  bool parseSubscript(std::string *Out);
  std::unique_ptr<LERExpression> parseExpression();
  std::unique_ptr<LERExpression>
  parseBinaryOpExpression(std::unique_ptr<LERExpression> LHS,
                          LEROperatorPrecedence Prec);
  std::unique_ptr<LERExpression> parseResult();

public:
  LERParser(std::unique_ptr<LERLexer> Lexer) : Lexer(std::move(Lexer)) {
    // Load the first token.
    advance();
  }

  // Helper method to invoke the Lexer across the entire input.
  void lexAndPrintTokens();

  MemoryBufferRef getSourceRef() const {
    return Lexer->LERInputBuffer->getMemBufferRef();
  }
};

} // namespace ler
#endif // LERIR_FRONTEND_H
