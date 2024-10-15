// Passes.td
// ~~~~~~~~~
// All LER transformation pass definitions
#ifndef LER_TRANSFORMS
#define LER_TRANSFORMS

include "mlir/Pass/PassBase.td"

def InjectInductionVars : Pass<"inject-induction-vars", "mlir::ModuleOp"> {
  let summary =
      "Replaces ler.Variable references with their respective loop index variables (block arguments).";
}

#endif // LER_TRANSFORMS
