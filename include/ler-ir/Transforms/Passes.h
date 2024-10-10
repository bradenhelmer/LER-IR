// Passes.h
// ~~~~~~~~
// LER Transform pass header.
#ifndef LER_TRANSFORM_PASSES_H
#define LER_TRANSFORM_PASSES_H

#include <mlir/Pass/Pass.h>

namespace ler {

#define GEN_PASS_DECL
#include <ler-ir/Transforms/Passes.h.inc>

#define GEN_PASS_REGISTRATION
#include <ler-ir/Transforms/Passes.h.inc>

} // namespace ler

#endif // LER_TRANSFORM_PASSES_H
