# `ler-compile` Internals Guide

This document is aimed for future developers looking to gain an understanding on the internals
of how this tool works. It is not a huge codebase, but there many steps taken, along with the 
help of LLVM and MLIR, to lower LER notation into a targeted, working, executable.

The `main` function is in [LERMain.cpp](../src/LERMain.cpp). In order, the stages of compilation are:
- [Frontend](#ler-frontend)
- [MLIR Code Generation](#mlir-code-generation)
- [MLIR Transformations and Conversions](#mlir-transformations-%26-conversions)
- [LLVM-IR, Assembly, and Executable Generation](#llvm-ir%2C-assembly%2C-and-executable-generation)

## LER Frontend
### Overview
The frontend of the compiler is the first stage in the compilation pipeline. This stage handles parsing
LER notation into an abstract syntax tree to be passed onto the code generator stage.

### LER Notation
As described in the paper, LER notation is a mathematical representation used to describe loop nests
in a traditional C program. This notation can be formally described using a context free grammar. The
grammar expected by the compiler can be found [here](../grammar). For a clear understanding of the 
notation, it would be best to read the paper.

**Note**: The grammar expected from the compiler is not a 1 to 1 representation of the notation described
in the paper. This is due to the use of mathematical notation that isn't suited for ASCII parsing.

Here is some example notation:

    ^Ri|1,M|^Sk|0,i|^Sj|0,i|x[i,j] * y[j,k] = r[i]

This maps to the equivalent C code like so:

```c
for (int i = 1; i < M; ++i)
{
    for (int k = 0; k < i; ++k)
    {
        for (int j = 0; j < i; ++j)
        {
            r[i] = x[i][j] * y[j][k];
        }
    }
}
```

### Source Files
The core files of the LER frontend are:
- [LERFrontend.h](../include/ler-ir/LERFrontend.h) - Header file for all Frontend classes, notably the `LERToken`, `LERLexer`, `LERParser`, and all AST types.
- [LERFrontend.cpp](../src/LERFrontend.cpp) - Implementation file for mostly `LERLexer` and `LERParser` methods.
- [LERTokenDefs.h](../include/ler-ir/LERTokenDefs.h) - Token kind macro definitions.

### Implementation Notes
- The `LERLexer` is implemented as any classical lexer, with the no public methods as it is only used by the
`LERParser`, hence the `friend class LERParser` in the class.
-  The `LERParser` has a method for each non-terminal rule in the grammar. Most of these methods return
a `unique_ptr` of some AST class.
- The `parseLERTree()` method is the core routine to parse an LER AST, this is called from the main function in [LERMain.cpp](../src/LERMain.cpp) and returns an `LERTree` class.
- `LERParser::lexAndPrintTokens()` is useful for debugging purposes.
- The command line option `--print-ast` will print the AST out. The above example will print:
```
LER AST for source: ^Ri|1,M|^Sk|0,i|^Sj|0,i|x[i,j] * y[j,k] = r[i]
    ForLoop: REGULAR_FOR i=1->M
      ForLoop: SUMMATION k=0->i
        ForLoop: SUMMATION j=0->i
           x[i,j] * y[j,k] = r[i]
```

## MLIR Code Generation
### Overview
After the frontend has parsed the LER notation into an AST, stored in a `LERTree` class, this AST is lowered into a
`mlir::ModuleOp` containing operations from the `LERDialect`.

The `LERDialect` is defined using TableGen in [LERDialect.td](../include/ler-ir/IR/LERDialect.td). Each structure in 
the grammar is essentially mapped to a custom operation in the dialect. For simplicity, each Operation returning an
`mlir::Value` returns an `I64` type. There is auto-generated documentation for all the operations [here](./LERDialect.md). The LER MLIR generated for the above example is:
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
        }) : () -> ()
      }) : () -> ()
    }) : () -> ()
  }) : () -> ()
}) {ler.Source = "^Ri|1,M|^Sk|0,i|^Sj|0,i|x[i,j] * y[j,k] = r[i]\0A\0A"} : () -> ()
```


### Source Files
- [LERDialect.td](../include/ler-ir/IR/LERDialect.td) - TableGen dialect definitions.
- [LERDialect.h](../include/ler-ir/IR/LERDialect.h) - Various global includes for the `LERDialect`.
- [LERDialect.cpp](../src/IR/LERDialect.cpp) - Various dialect specific method implementations for things llike custom builders and utility functions.
- [LERCodeGenerators.cpp](../src/LERCodeGenerators.cpp) - virtual `LERASTNode::codeGen()` implementations for each AST class.

### Implementation Notes
- Only `LERExpression` derived AST classes return `mlir::Value`
- A `mlir::func::FuncOp` encapsulates the entire loop nest as the `main` function for lowering.

## MLIR Transformations & Conversions
### Overview
The goal of this stage is to lower the module of LER operations fully into the `LLVMDialect`. There are 
currently 5 passes to achieve this using intermediary dialects.
Similar to the LER dialect and operations, there is auto-generated documentation
on the purpose of thsese passes that can be found [here](./LERPasses.md).
### Source Files (Descriptions in auto-generated docs).
- [Passes.td](../include/ler-ir/Passes.td) - TableGen pass definitions.
- [Passes.h](../include/ler-ir/Passes.h) - Pass header file.
- [InjectInductionVars.cpp](../src/Transformations/InjectInductionVars.cpp)
- [ConvertArrayAccToMemref.cpp](../src/Conversion/ConvertArrayAccToMemref.cpp)
- [ConvertLoopsToAffineSCF.cpp](../src/Conversion/ConvertLoopsToAffineSCF.cpp)
- [ConvertToArith.cpp](../src/Conversion/ConvertToArith.cpp)
- [ConvertToLLVM.cpp](../src/Conversion/ConvertToLLVM.cpp)
- [Misc.h](../include/ler-ir/Analysis/Misc.h) - Miscellaneous analysis definitions.
- [Misc.cpp](../src/Analysis/Misc.cpp) - Miscellaneous analysis implementations.

### Implementation Notes
- Each pass is given its own source file.
- Some of the passes could be implemented a lot cleaner, particularly `ConvertLoopsToAffineSCF`.

## LLVM-IR, Assembly, and Executable Generation
### Overview
This is the final stage in the compiler pipeline, invoked by passing the compiler the `--exe` option.
### Source Files
- [LERUtils.h](../include/ler-ir/LERUtils.h) - Defines the `moduleToExecutable` method.
- [LERUtils.cpp](../src/LERUtils.cpp) - Implements the `moduleToExecutable` method.
### Implementation Notes
- The `moduleToExecutable` is a three stage method to lower the lowered `LLVMDialect` into an executable:

    1. Translate the `LLVMDialect` into LLVM-IR
    2. Use the `llc` tool from the LLVM toolchain to lower LLVM-IR into targetable assembly.
    3. Use the system compiler (clang | gcc) to assemble into an executable.

- The intermediary LLVM and assembly files can be kept with command line options. See [USAGE.md.](./USAGE.md)
