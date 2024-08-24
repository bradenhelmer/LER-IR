// LEROps.td
// ~~~~~~~~~
// LER operation definitions.
#ifndef LER_OPS
#define LER_OPS

include "LERBase.td"
include "mlir/IR/OpBase.td"

// Base LER operation class.
class LEROp<string mnemonic, list<Trait> traits = []> :
    Op<LER_Dialect, mnemonic, traits>;

class LER_Loop<string mnemonic, list<Trait> traits = []> :
	LEROp<mnemonic, traits> {

	let summary = "Base operation for LER based loops.";

	let regions = (
		region SizedRegion<1>:$nest
	);
}

class LER_ForLoop<string mnemonic, list<Trait> traits = []> :
	LER_Loop<mnemonic, traits> {

	let summary = "Base operation for 'for' based loops.";

	let arguments = (
		ins Index:$lowerBound,
			Index:$upperBound,
			DefaultValueAttr<I64Attr, "1">:$step	
	);

}

def LER_RegularForLoop : LER_ForLoop<"for"> {}
def LER_SummationLoop : LER_ForLoop<"summation"> {}
def LER_ProductLoop : LER_ForLoop<"product"> {}
def LER_WhileLoop : LER_Loop<"while"> {}

#endif
