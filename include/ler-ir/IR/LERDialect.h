// LERDialect.h
// ~~~~~~~~~~~~
// Core LER dialect header.
#ifndef LERIR_LER_DIALECT_H
#define LERIR_LER_DIALECT_H
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>

#include <ler-ir/IR/LERDialect.h.inc>
#include <mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/ParallelCombiningOpInterface.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>

#define GET_OP_CLASSES
#include <ler-ir/IR/LEROps.h.inc>
#endif
