// Passes.h
// ~~~~~~~~
// LER pass header.
#ifndef LER_PASSES_H
#define LER_PASSES_H

#include <ler-ir/IR/LERDialect.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace ler {

#define GEN_PASS_DECL
#include <ler-ir/Passes.h.inc>

#define GEN_PASS_REGISTRATION
#include <ler-ir/Passes.h.inc>

} // namespace ler

#define UNKNOWN_LOC ReWriter.getUnknownLoc()

#endif // LER_CONVERSION_PASSES_H
