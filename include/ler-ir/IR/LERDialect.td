// LEROps.td
// ~~~~~~~~~
// LER operation definitions.
// Alot of these operations, particularly the loops, are loosely
// based off of the SCF dialect.
#ifndef LER_OPS
#define LER_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LER_Dialect : Dialect {
	let name = "ler";
	let summary = "The LER dialect defines operations for representing LER notation.";

	let cppNamespace = "::ler";
	
	// Extended description
	let description = [{}];
}

// Base LER operation class.
class LER_Op<string mnemonic, list<Trait> traits = []> :
    Op<LER_Dialect, mnemonic, traits>;

// LOOP OPERATIONS
// ~~~~~~~~~~~~~~~

// Base operation for representing a loop in the LER dialect.
// This operation will derive into these 4 actual loops:
// 	1. Regular for loop
// 	2. Summation for loop
// 	3. Product for loop
// 	4. While loop
class LER_LoopOp<string mnemonic, list<Trait> traits = [
    DeclareOpInterfaceMethods<LoopLikeOpInterface>
]> :
	LER_Op<mnemonic, traits> {
	let regions = (region AnyRegion:$Region);
}

// While loop operation, takes a bit for a condition.
def LER_WhileLoopOp : LER_LoopOp<"While"> {
	let arguments = (ins I1:$Condition);
}

// Base class for 'For Loop' operations.
class LER_ForLoopOp<string mnemonic> :
	LER_LoopOp<mnemonic> {

	let arguments = (ins AnySignlessInteger:$LowerBound,
						 AnySignlessInteger:$UpperBound,
						 AnySignlessInteger:$Step);

	let builders = [
    	OpBuilder<(ins "int64_t":$LowerBound, 
					   "int64_t":$UpperBound,
	                   CArg<"int64_t", "1">:$Step)>,
	];
}

// Actual For Loop Ops
def LER_RegularForLoopOp : LER_ForLoopOp<"RegularFor">;
def LER_SummationForLoopOp : LER_ForLoopOp<"Summation">;
def LER_ProductionForLoopOp : LER_ForLoopOp<"Production">;


// EXPRESSION OPERATIONS
// ~~~~~~~~~~~~~~~~~~~~~
/*
* The LER dialect is interested in maintaining a high level
* view over the loop nest. Consequently, the core sub expressions
* of an LER expression that concerns this dialect are:
* 	1. Constants e.g 1 | 100
* 	2. Variables e.g a | b
* 	3. Array accesses e.g x[i,j] | y[i, j ,k]
* 	4. Function calls e.g sin(x)
* 	5. A combination of these piece
*/

def LER_ExpressionOp : LER_Op<"Expression",
	[ParentOneOf<["RegularForLoopOp", "SummationForLoopOp", "ProductionForLoopOp", "WhileLoopOp"]>]> {
	let arguments = (ins Variadic<AnyType>:$Operands);
	let hasVerifier = 1;
}

class LER_BinaryExpressionOp<string mnemonic, list<Trait> traits =
	[HasParent<"ExpressionOp">]>
	: LER_Op<mnemonic, traits> {
	let arguments = (ins AnyType:$LHS, AnyType:$RHS);
	let results = (outs AnyType:$Result);
}

def LER_AddOp : LER_BinaryExpressionOp<"Add">;
def LER_SubOp : LER_BinaryExpressionOp<"Sub">;
def LER_MulOp : LER_BinaryExpressionOp<"Mul">;
def LER_DivOp : LER_BinaryExpressionOp<"Div">;

class LER_SubExpressionOp<string mnemonic, list<Trait> traits =
	[ParentOneOf<["ExpressionOp", "AddOp", "SubOp", "MulOp", "DivOp"]>]>
	: LER_Op<mnemonic, traits>;

def LER_ConstantOp : LER_SubExpressionOp<"Constant", [Pure]> {
	let arguments = (ins AnySignlessInteger:$Value);
	let results = (outs AnySignlessInteger:$Result); 
}

def LER_VariableOp: LER_SubExpressionOp<"Variable"> {
	let arguments = (ins SymbolNameAttr:$Name);
	let results = (outs AnyType:$Result);
}

def LER_ArrayAccessOp : LER_SubExpressionOp<"ArrayAccess"> {
	let arguments = (ins SymbolNameAttr:$ArrayName, Variadic<Index>:$Indicies);
	let results = (outs AnyType:$Result);
}
def LER_FunctionCallOp : LER_SubExpressionOp<"FunctionCall"> {
	let arguments = (ins SymbolNameAttr:$FunctionName, Variadic<AnyType>:$Parameters);
	let results = (outs AnyType:$Result);
}

#endif
