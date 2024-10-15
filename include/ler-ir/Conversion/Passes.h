// Passes.h
// ~~~~~~~~
// LER conversion pass header.
#ifndef LER_CONVERSION_PASSES_H
#define LER_CONVERSION_PASSES_H

#include <ler-ir/IR/LERDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace ler {

#define GEN_PASS_DECL
#include <ler-ir/Conversion/Passes.h.inc>

#define GEN_PASS_REGISTRATION
#include <ler-ir/Conversion/Passes.h.inc>

} // namespace ler

#endif // LER_CONVERSION_PASSES_H
