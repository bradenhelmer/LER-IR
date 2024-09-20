// LERDialect.h
// ~~~~~~~~~~~~
// Core LER dialect header.
#ifndef LERIR_LER_DIALECT_H
#define LERIR_LER_DIALECT_H
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

#include <ler-ir/IR/LERDialect.h.inc>
#define GET_OP_CLASSES
#include <ler-ir/IR/LEROps.h.inc>
#endif
