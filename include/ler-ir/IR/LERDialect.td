// LEROps.td
// ~~~~~~~~~
// LER operation definitions.
// Alot of these operations, particularly the loops, are loosely
// based off of the SCF dialect.
#ifndef LER_OPS
#define LER_OPS

include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def LER_Dialect : Dialect {
  let name = "ler";
  let summary =
      "The LER dialect defines operations for representing LER notation.";

  let cppNamespace = "::ler";
}

class LER_Op<string mnemonic, list<Trait> traits = []>
    : Op<LER_Dialect, mnemonic, traits>;

class LER_LoopOp<string mnemonic, list<Trait> traits = []>
    : LER_Op<mnemonic, !listconcat(
                           [
                             Terminator, SymbolTable,
                             DeclareOpInterfaceMethods<LoopLikeOpInterface>
                           ],
                           traits)> {
  let regions = (region SizedRegion<1> : $Region);
}

def LER_WhileLoopOp : LER_LoopOp<"While"> {
  let arguments = (ins OptionalAttr<StrAttr>:$Condition,
						 OptionalAttr<StrArrayAttr>:$Subscripts);
  let builders =  [
		OpBuilder<(ins CArg<"llvm::StringRef", "\"\"">:$Condition,
					   CArg<"llvm::ArrayRef<std::string>", "{}">:$Subscripts)>,
	];
}

class LER_ForLoopOp<string mnemonic>
    : LER_LoopOp<mnemonic, [DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {

  let arguments = (ins AnyAttrOf<[I64Attr, SymbolRefAttr]>:$LowerBound,
                   AnyAttrOf<[I64Attr, SymbolRefAttr]>:$UpperBound,
                   SymbolNameAttr:$LoopIdxVar,
                   DefaultValuedAttr<I64Attr, "1">:$Step);

  let builders = [
    OpBuilder<(ins "std::string":$LowerBound, 
                  "std::string":$UpperBound,
                  "std::string":$LoopIdxVar),
    [{
      auto BoundAttrs = 
        ler::convertForLoopBoundsToAttrs($_builder, LowerBound, UpperBound);        
      build($_builder, $_state, BoundAttrs.first, BoundAttrs.second, LoopIdxVar);
    }]>
  ];
}

def LER_RegularForLoopOp : LER_ForLoopOp<"RegularFor">;
def LER_SummationForLoopOp : LER_ForLoopOp<"Summation">;
def LER_ProductionForLoopOp : LER_ForLoopOp<"Production">;

class LER_SubExpressionOp<string mnemonic, list<Trait> traits =
	[]>: LER_Op<mnemonic, traits>;

class LER_BinaryExpressionOp<string mnemonic, list<Trait> traits =[] >
	: LER_SubExpressionOp<mnemonic, traits> {
	let arguments = (ins AnyType:$LHS, AnyType:$RHS);
	let results = (outs I64:$Result);
	let builders = [
		OpBuilder<(ins "mlir::Value":$LHS, "mlir::Value":$RHS), [{
		build($_builder, $_state, $_builder.getI64Type(), LHS, RHS);
		}]>
	];
}

def LER_AddOp : LER_BinaryExpressionOp<"Add">;
def LER_SubOp : LER_BinaryExpressionOp<"Sub">;
def LER_MulOp : LER_BinaryExpressionOp<"Mul">;
def LER_DivOp : LER_BinaryExpressionOp<"Div">;

def LER_ConstantOp : LER_SubExpressionOp<"Constant", [Pure]> {
	let arguments = (ins I64Attr:$Value);
	let results = (outs I64:$Result);
	let builders = [
		OpBuilder<(ins "int64_t":$Value), [{
		build($_builder, $_state, $_builder.getI64Type(),
			  $_builder.getI64IntegerAttr(Value));
		}]>
	];
}

def LER_VariableOp: LER_SubExpressionOp<"Variable"> {
	let arguments = (ins SymbolRefAttr:$Name, OptionalAttr<StrArrayAttr>:$Subscripts);
	let results = (outs I64:$Result);
	let builders = [
		OpBuilder<(ins "std::string":$Name, "llvm::ArrayRef<std::string>":$Subscripts)>
	];
	let extraClassDeclaration = [{
		::llvm::StringRef getNameAsStrRef() {
			return getNameAttr().getLeafReference().getValue();
		}
	}];
}

def LER_ArrayAccessOp : LER_SubExpressionOp<"ArrayAccess"> {
  let arguments = (ins SymbolRefAttr : $ArrayName, Variadic<I64> : $Indicies);
  let results = (outs I64 : $Result);
}

def LER_FunctionCallOp : LER_SubExpressionOp<"FunctionCall"> {
  let arguments = (ins SymbolRefAttr:$FunctionName, Variadic<AnyType>:$Parameters);
  let results = (outs I64 : $Result);
}

def LER_ParenExprOp : LER_SubExpressionOp<"ParenExpr"> {
  let arguments = (ins I64 : $Expression);
  let results = (outs I64 : $Result);
}

def LER_ResultOp : LER_Op<"Result", [Terminator]> {
  let arguments = (ins AnyType : $Expression, AnyType : $Location);
}

#endif
