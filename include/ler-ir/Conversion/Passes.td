// Passes.td
// ~~~~~~~~~
// LER conversion pass definitions.
#ifndef LER_CONVERSION
#define LER_CONVERSION

include "mlir/Pass/PassBase.td"

def ConvertArrayAccToMemref : Pass<"convert-array-acc-to-memref", "ler::ArrayAccessOp"> {
	let summary =
		"Creates actual memref arrays from ler.ArrayAccesses.";
	let dependentDialects = ["::mlir::memref::MemRefDialect"];
}

def ConvertForLoopsToAffine : Pass<"convert-for-to-affine"> {
	let summary =
		"Converts ler.[Production|Regular|Summartion]ForLoops to affine.for.";
}

def ConvertWhileLoopsToSCF : Pass<"convert-while-to-scf", "ler::WhileLoopOp"> {}

#endif // LER_CONVERSION
