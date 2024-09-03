// LEROps.td
// ~~~~~~~~~
// LER operation definitions.
#ifndef LER_OPS
#define LER_OPS

include "mlir/IR/OpBase.td"

def LER_Dialect : Dialect {
	let name = "ler";
	let summary = "The LER dialect defines operations for representing LER notation.";

	let cppNamespace = "::ler";

	// Use affine and SCF dialects for loop representations.
	let dependentDialects = [
		"mlir::affine::AffineDialect",
		"mlir::scf::SCFDialect"
	];
	
	// Extended description
	let description = [{}];
}

// Base LER operation class.
class LEROp<string mnemonic, list<Trait> traits = []> :
    Op<LER_Dialect, mnemonic, traits>;

def LER_Collection : LEROp<"collection"> {
	let summary = "Collection of LER formulae converted to various operations.";
	let regions = (region SizedRegion<1>:$collection);

	let hasRegionVerifier = 1;
}


#endif
