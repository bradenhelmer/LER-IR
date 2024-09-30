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
	
}

class LER_Op<string mnemonic, list<Trait> traits = []> :
    Op<LER_Dialect, mnemonic, traits>;

class LER_LoopOp<string mnemonic, list<Trait> traits = [
    DeclareOpInterfaceMethods<LoopLikeOpInterface>
]> :
	LER_Op<mnemonic, traits> {
	let regions = (region SizedRegion<1>:$Region);
}

def LER_WhileLoopOp : LER_LoopOp<"While"> {
	let arguments = (ins OptionalAttr<StrAttr>:$Condition,
						 OptionalAttr<StrArrayAttr>:$Subscripts);
	let builders =  [
		OpBuilder<(ins CArg<"llvm::StringRef", "\"\"">:$Condition,
					   CArg<"llvm::ArrayRef<std::string>", "{}">:$Subscripts)>,
	];
}

class LER_ForLoopOp<string mnemonic> :
	LER_LoopOp<mnemonic> {

	let arguments = (ins AnyAttrOf<[I64Attr, SymbolRefAttr]>:$LowerBound,
						 AnyAttrOf<[I64Attr, SymbolRefAttr]>:$UpperBound,
						 SymbolRefAttr:$LoopIdxVar,
						 DefaultValuedAttr<I64Attr, "1">:$Step);

	let builders = [
    	OpBuilder<(ins "std::string":$LowerBound, 
					   "std::string":$UpperBound,
					   "std::string":$LoopIdxVar), [{
		auto BoundAttrs = 
			ler::convertForLoopBoundsToAttrs($_builder, LowerBound, UpperBound);		
		auto LoopIdxVarAttr = 
			::mlir::SymbolRefAttr::get($_state.getContext(), LoopIdxVar);		
		build($_builder, $_state, BoundAttrs.first, BoundAttrs.second, LoopIdxVarAttr);
		}]>
	];
}

def LER_RegularForLoopOp : LER_ForLoopOp<"RegularFor">;
def LER_SummationForLoopOp : LER_ForLoopOp<"Summation">;
def LER_ProductionForLoopOp : LER_ForLoopOp<"Production">;

def LER_ExpressionOp : LER_Op<"Expression",
	[ParentOneOf<["RegularForLoopOp", "SummationForLoopOp", "ProductionForLoopOp", "WhileLoopOp"]>]> {
	let arguments = (ins Variadic<AnyType>:$Operands);
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
