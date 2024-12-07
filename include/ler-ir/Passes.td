// Passes.td
// ~~~~~~~~~
// LER pass defintions.
#ifndef LER_PASSES
#define LER_PASSES

include "mlir/Pass/PassBase.td"

// ANALYSIS

// CONVERSION

def ConvertArrayAccToMemref
    : Pass<"convert-array-acc-to-memref", "mlir::ModuleOp"> {
  let summary = "Creates actual memref arrays from ler.ArrayAccesses.";
  let description = [{
	Due to LER notation containing array accesses that don't have any corresponding
	array declarations to actually access, these must be created using
	`memref.Alloc` operations. The sizes of these arrays will never be more
	than 6 dimensions, or more than 1 million total items. For example, an
	access that is one dimensional like `x[i]` will have an array  created like so:

		%0 = memref.alloc() : memref<100000xi64>	
	Next, the `ler.ArrayAccess` operations will be lowered into `memref.Load`
	operations. A module before this pass may look like:
	```mlir
	"builtin.module"() <{sym_name = "../test/case10.ler"}> ({
		"func.func"() <{function_type = () -> (), sym_name = "main"}> ({
		"ler.RegularFor"() <{LoopIdxVar = "i", LowerBound = 1 : i64, Step = 1 : i64, UpperBound = @M}> ({
		^bb0(%arg0: index):
		  "ler.Summation"() <{LoopIdxVar = "k", LowerBound = 0 : i64, Step = 1 : i64, UpperBound = @i}> ({
		  ^bb0(%arg1: index):
			"ler.Summation"() <{LoopIdxVar = "j", LowerBound = 0 : i64, Step = 1 : i64, UpperBound = @i}> ({
			^bb0(%arg2: index):
			  %0 = "ler.Variable"() <{Name = @i}> : () -> i64
			  %1 = "ler.Variable"() <{Name = @j}> : () -> i64
			  %2 = "ler.ArrayAccess"(%0, %1) <{ArrayName = @x}> : (i64, i64) -> i64
			  %3 = "ler.Variable"() <{Name = @j}> : () -> i64
			  %4 = "ler.Variable"() <{Name = @k}> : () -> i64
			  %5 = "ler.ArrayAccess"(%3, %4) <{ArrayName = @y}> : (i64, i64) -> i64
			  %6 = "ler.Mul"(%2, %5) : (i64, i64) -> i64
			  %7 = "ler.Variable"() <{Name = @i}> : () -> i64
			  %8 = "ler.ArrayAccess"(%7) <{ArrayName = @r}> : (i64) -> i64
			  "ler.Result"(%6, %8) : (i64, i64) -> ()
			  "ler.LoopTerminator"() : () -> ()
			}) : () -> ()
			"ler.LoopTerminator"() : () -> ()
		  }) : () -> ()
		  "ler.LoopTerminator"() : () -> ()
		}) : () -> ()
		"func.return"() : () -> ()
	  }) : () -> ()
	}) {ler.Source = "^Ri|1,M|^Sk|0,i|^Sj|0,i|x[i,j]*y[j,k]=r[i]\0A"} : () -> ()
	```
	After:
	```mlir
	module @case10.conv.ler attributes {ler.Source = "^Ri|1,M|^Sk|0,i|^Sj|0,i|x[i,j]*y[j,k]=r[i]\0A"} {
	  func.func @main() {
		%alloc = memref.alloc() : memref<1000000xi64>
		%alloc_0 = memref.alloc() : memref<1000x1000xi64>
		%alloc_1 = memref.alloc() : memref<1000x1000xi64>
		"ler.RegularFor"() <{LoopIdxVar = "i", LowerBound = 1 : i64, Step = 1 : i64, UpperBound = @M}> ({
		^bb0(%arg0: index):
		  "ler.Summation"() <{LoopIdxVar = "k", LowerBound = 0 : i64, Step = 1 : i64, UpperBound = @i}> ({
		  ^bb0(%arg1: index):
			"ler.Summation"() <{LoopIdxVar = "j", LowerBound = 0 : i64, Step = 1 : i64, UpperBound = @i}> ({
			^bb0(%arg2: index):
			  %0 = memref.load %alloc_1[%arg0, %arg2] : memref<1000x1000xi64>
			  %1 = memref.load %alloc_0[%arg2, %arg1] : memref<1000x1000xi64>
			  %2 = arith.muli %0, %1 : i64
			  %3 = memref.load %alloc[%arg0] : memref<1000000xi64>
			  "ler.Result"(%2, %3) : (i64, i64) -> ()
			  "ler.LoopTerminator"() : () -> ()
			}) : () -> ()
			"ler.LoopTerminator"() : () -> ()
		  }) : () -> ()
		  "ler.LoopTerminator"() : () -> ()
		}) : () -> ()
		return
	  }
	}
	```
	**NOTE**: This pass assumes the the `-inject-induction-vars` pass has ran before.
  }];
  let dependentDialects = ["::mlir::memref::MemRefDialect"];
}

def ConvertLoopsToAffineSCF : Pass<"convert-loops-to-affine-scf", "mlir::ModuleOp"> {
  let summary =
      "Converts ler.[Production|Regular|Summartion]ForLoops to affine.for. "
      "This pass also converts ler.WhileLoops to SCF.while.";
  let description = [{
	This pass takes LER loop operations and lowers them into intermediary dialects
	before into LLVM. All of the LER for loop operatons (Production|Regular|Summartion)
	are lowered into `affine.For` operations and `ler.WhileLoop` are lowered into
	`scf.While` operations. When lowering into `affine.For` operations, affine mappings
	are created during runtime to ensure all of the created arrays are not accessed
	out of bounds. When converting `ler.While` operations, a simple counter is implemented
	to act as the condition for the loop, giving a constant iteration count. This pass 
	also converts `ler.Result` ops to `memref.Store` and `ler.Variable` ops to 
	`arith.Constant`. Module before:

	```mlir
	module @case10.conv.ler attributes {ler.Source = "^Ri|1,M|^Sk|0,i|^Sj|0,i|x[i,j]*y[j,k]=r[i]\0A"} {
	  func.func @main() {
		%alloc = memref.alloc() : memref<1000000xi64>
		%alloc_0 = memref.alloc() : memref<1000x1000xi64>
		%alloc_1 = memref.alloc() : memref<1000x1000xi64>
		"ler.RegularFor"() <{LoopIdxVar = "i", LowerBound = 1 : i64, Step = 1 : i64, UpperBound = @M}> ({
		^bb0(%arg0: index):
		  "ler.Summation"() <{LoopIdxVar = "k", LowerBound = 0 : i64, Step = 1 : i64, UpperBound = @i}> ({
		  ^bb0(%arg1: index):
			"ler.Summation"() <{LoopIdxVar = "j", LowerBound = 0 : i64, Step = 1 : i64, UpperBound = @i}> ({
			^bb0(%arg2: index):
			  %0 = memref.load %alloc_1[%arg0, %arg2] : memref<1000x1000xi64>
			  %1 = memref.load %alloc_0[%arg2, %arg1] : memref<1000x1000xi64>
			  %2 = arith.muli %0, %1 : i64
			  %3 = memref.load %alloc[%arg0] : memref<1000000xi64>
			  "ler.Result"(%2, %3) : (i64, i64) -> ()
			  "ler.LoopTerminator"() : () -> ()
			}) : () -> ()
			"ler.LoopTerminator"() : () -> ()
		  }) : () -> ()
		  "ler.LoopTerminator"() : () -> ()
		}) : () -> ()
		return
	  }
	}
	```
	Module after:
	```mlir
	module @case10.conv.ler attributes {ler.Source = "^Ri|1,M|^Sk|0,i|^Sj|0,i|x[i,j]*y[j,k]=r[i]\0A"} {
	  func.func @main() {
		%alloc = memref.alloc() : memref<1000000xi64>
		%alloc_0 = memref.alloc() : memref<1000x1000xi64>
		%alloc_1 = memref.alloc() : memref<1000x1000xi64>
		affine.for %arg0 = 1 to 1000 {
		  affine.for %arg1 = 0 to 1000 {
			affine.for %arg2 = 0 to 1000 {
			  %0 = memref.load %alloc_1[%arg0, %arg2] : memref<1000x1000xi64>
			  %1 = memref.load %alloc_0[%arg2, %arg1] : memref<1000x1000xi64>
			  %2 = arith.muli %0, %1 : i64
			  %3 = memref.load %alloc[%arg0] : memref<1000000xi64>
			  memref.store %2, %alloc[%arg0] : memref<1000000xi64>
			}
		  }
		}
		return
	  }
	}
	```

	**NOTE**: This pass assumes the the `-inject-induction-vars` pass has ran before.


  }];
  let dependentDialects = ["::mlir::affine::AffineDialect", "::mlir::scf::SCFDialect"];
}

def ConvertToArith : Pass<"convert-to-arith", "mlir::ModuleOp"> {
  let summary = "Converts existing ler.Variables to arith.Constants and binary "
                "ops to their respective"
                "arith counterparts";
  let description = [{}];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

def ConvertToLLVM : Pass<"convert-to-llvm", "mlir::ModuleOp"> {
  let summary = "Converts all intermediary dialects (affine, arith, memref, scf) to llvm";
  let description = [{}];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"]; 
}

// TRANSFORMS

def InjectInductionVars : Pass<"inject-induction-vars", "mlir::ModuleOp"> {
  let summary = "Replaces ler.Variable references with their respective loop "
                "index variables (block arguments).";
  let description = [{}];
}

#endif
