// Frontend.cpp
// ~~~~~~~~~~~~
// Token, Lexer, and Parser implementations.
#include <iostream>
#include <ler-ir/LERCommonUtils.h>
#include <ler-ir/LERFrontend.h>

namespace ler {

LERLexer::LERLexer(const std::string &InputFilename) {

  auto InputBufferOrErr = llvm::MemoryBuffer::getFile(InputFilename, true);

  if (!InputBufferOrErr) {
    ERRS << "There was an error reading '" << InputFilename
         << "'!\nError: " << InputBufferOrErr.getError().message();
  }

  LERInputBuffer = std::move(*InputBufferOrErr);
  LERBufStart = LERInputBuffer->getBufferStart();
  LERBufEnd = LERInputBuffer->getBufferEnd();
  LERBufCurr = LERBufStart;
}

bool LERLexer::lexToken(LERToken *Out) {

  if (isWhiteSpace(*LERBufCurr)) {
    do {
      LERBufCurr++;
    } while (isWhiteSpace(*LERBufCurr));
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
  while (Lexer->lexToken(&CurrToken)) {
    OUTS << TokenNames[CurrToken.Kind] << '\n';
    if (CurrToken.Kind == LER_EOF)
      break;
  }
  Lexer->LERBufCurr = Lexer->LERBufStart;
}

} // namespace ler
