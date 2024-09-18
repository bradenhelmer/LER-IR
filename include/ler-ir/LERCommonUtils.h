// CommonUtils.h
// ~~~~~~~~~~~~~
// Common macro definitions and helper functions.
#ifndef LERIR_COMMON_UTILS_H
#define LERIR_COMMON_UTILS_H
#include <llvm/Support/raw_ostream.h>

#define OUTS llvm::outs()
#define ERRS llvm::errs()

#define PRINT_INDENT()                                                         \
  for (int i = 0; i < Indent; ++i) {                                           \
    OUTS << "  ";                                                              \
  }

#define PRINT_SUBSCRIPTS()                                                     \
  if (Subscripts.size() > 0) {                                                 \
    OUTS << "Subscripts (";                                                    \
    auto SubscriptCount = getSubscriptCount();                                 \
    for (int i = 0; i < SubscriptCount; ++i) {                                 \
      OUTS << Subscripts[i];                                                   \
      if (i < SubscriptCount - 1) {                                            \
        OUTS << ',';                                                           \
      }                                                                        \
    }                                                                          \
    OUTS << ")";                                                               \
  }

#endif
