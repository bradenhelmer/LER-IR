// LERBase.td
// ~~~~~~~~~~
// Base definitions for the LER dialect.
#ifndef LER_BASE
#define LER_BASE

include "mlir/IR/DialectBase.td"

// Base LER dialect defintion.
def LER_Dialect : Dialect {
	let name = "ler";
	let summary = "The LER dialect defines operations for representing LER notation.";

	let cppNamespace = "::ler";

	// We can use the affine dialect's AffineForOp, as well as
	// the SCF dialect's WhileOp for.
	
	//let dependentDialects = [
	//	"mlir::affine::AffineDialect",
	//	"mlir::scf::SCFDialect"
	//];

	// Extended description
	let description = [{}];
}

#endif
