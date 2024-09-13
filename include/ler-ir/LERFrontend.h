// LERFrontend.h
// ~~~~~~~~~~
// Token, Lexer, and Parser definitions.
#ifndef LERIR_FRONTEND_H
#define LERIR_FRONTEND_H
#include <llvm/Support/MemoryBuffer.h>

namespace ler {

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

struct LERToken {
  const char *Start;
  const char *End;
  LERTokenKind Kind;
};

class LERLexer {

  // Only the parser will need access to the lexer's members.
  friend class LERParser;

  // LER input file.
  std::unique_ptr<llvm::MemoryBuffer> LERInputBuffer;

  // Buffer pointers
  const char *LERBufStart;
  const char *LERBufEnd;
  const char *LERBufCurr;

  // Lexical methods
  bool lexToken(LERToken *Out);
  bool lexIdentifier(LERToken *Out);
  bool lexLoopIdentifier(LERToken *Out);
  bool lexNumber(LERToken *Out);

  // 'is' methods
  static inline bool isWhiteSpace(char c) {
    return c == ' ' | c == '\t' | c == '\r';
  }
  static inline bool isIdentifierChar(char c) { return isalpha(c) || c == '_'; }

public:
  LERLexer(const std::string &InputFilename);
};

class LERParser {
  LERToken CurrToken;

  std::unique_ptr<LERLexer> Lexer;

public:
  LERParser(std::unique_ptr<LERLexer> Lexer) : Lexer(std::move(Lexer)) {}

  void lexAndPrintTokens();
};

} // namespace ler
#endif // LERIR_FRONTEND_H
