// Passes.td
// ~~~~~~~~~
// LER pass defintions.
#ifndef LER_PASSES
#define LER_PASSES

include "mlir/Pass/PassBase.td"

// ANALYSIS

// CONVERSION

def ConvertArrayAccToMemref : Pass<"convert-array-acc-to-memref", "mlir::ModuleOp"> {
	let summary =
		"Creates actual memref arrays from ler.ArrayAccesses.";
	let dependentDialects = ["::mlir::memref::MemRefDialect"];
}

def ConvertForLoopsToAffine : Pass<"convert-for-to-affine", "mlir::ModuleOp"> {
	let summary =
		"Converts ler.[Production|Regular|Summartion]ForLoops to affine.for.";
	let dependentDialects = ["::mlir::affine::AffineDialect"];
}

def ConvertWhileLoopsToSCF : Pass<"convert-while-to-scf", "mlir::ModuleOp"> {
	let summary =
		"Converts ler.WhileLoops to scf.While";
	let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def ConvertToArith : Pass<"convert-to-arith", "mlir::ModuleOp"> {
	let summary = 
		"Converts existing ler.Variables to arith.Constants and binary ops to their respective"
		"arith counterparts";
	let dependentDialects = ["::mlir::arith::ArithDialect"];
}

// TRANSFORMS

def InjectInductionVars : Pass<"inject-induction-vars", "mlir::ModuleOp"> {
  let summary =
      "Replaces ler.Variable references with their respective loop index variables (block arguments).";
}

#endif
