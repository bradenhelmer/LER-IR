# `ler-compile` Internals Guide

This document is aimed for future developers looking to gain an understanding on the internals
of how this tool works. It is not a huge codebase, but there many steps taken, along with the
help of LLVM and MLIR, to lower LER notation into a targeted, working, executable.

The `main` function is in [LERMain.cpp](../src/LERMain.cpp). In order, the stages of compilation are:
- [GLORE Algorithm Background](#glore-algorithm-background)
- [Worked Example: Category 3 Redundancy](#worked-example-category-3-redundancy-on3--on2)
- [Frontend](#ler-frontend)
- [MLIR Code Generation](#mlir-code-generation)
- [MLIR Transformations and Conversions](#mlir-transformations-%26-conversions)
- [LLVM-IR, Assembly, and Executable Generation](#llvm-ir%2C-assembly%2C-and-executable-generation)
- [Java Optimizer Integration](#java-optimizer-integration)
- [Known Limitations](#known-limitations)

## GLORE Algorithm Background

`ler-compile` implements the GLORE algorithm from Ding & Shen (OOPSLA '17). GLORE detects and
removes loop redundancies by operating on a symbolic representation of loop nests called
LER-notation, rather than on the raw loop structure. This gives it significantly broader
applicability than traditional loop-invariant code motion (LICM), which only hoists scalar
expressions invariant to the innermost loop.

### The Four Redundancy Categories

GLORE classifies loop redundancies into four categories:

| Category | Description | Visibility to LICM |
| --- | --- | --- |
| 1 | Loop-invariant expression — same value every iteration | Yes |
| 2 | Partially loop-invariant expression — same value in some iterations | Rarely |
| 3 | Loop-invariant loop — entire nested loop recomputes the same result across outer loop iterations | No |
| 4 | Partially loop-invariant loop — nested loop's computation is a strict subset of a later iteration's computation | No |

This implementation targets **categories 3 and 4**, with the primary focus on category 3 for
regular `for` loop nests. These are the cases invisible to all traditional methods: LICM
works on scalar values at a single loop level; it has no mechanism to recognize that an entire
inner loop nest is being redundantly re-executed.

### LER-Notation: `L E ⇒ R`

Every loop nest is expressed as one or more LER-formulae of the form:

    L E ⇒ R

where `L` is a sequence of loop notations, `E` is the expression in the innermost loop body,
and `R` is the result variable being written. Loop types in `L`:

- **`^R` (regular for loop)** — `^Rx|l,u|` iterates x from l to u, stepping by 1. Bounds are
  affine expressions over enclosing loop indices.
- **`^S` (summation reduction)** — same structure as `^R` but semantically accumulates
  (reduces with `+`) across iterations.
- **`^P` (production reduction)** — same as `^S` but reduces with `*`.
- **`^W` (while loop)** — no explicit index variable; identified by a unique subscript ID
  tracking cross-iteration dependencies.

Operands carry optional **dependence subscripts** when their value flows from an outer loop
body into the inner loop, making data dependencies explicit in the notation rather than
implicit in the control flow.

### The `relLoops` Insight

The key concept enabling category 3 detection is `relLoops`. For any operand `x` in a
formula, `relLoops(x)` is the set of loop IDs that appear in `x`'s index expression — the
loops whose iteration actually affects the value of `x`. For a reduction loop `R`,
`relLoops(R)` is the union of `relLoops` over all operands of `R` that are themselves
relevant to `R`.

A loop `L₂` nested inside `L₁` is **loop-invariant with respect to `L₁`** if
`relLoops(L₂) ∩ {id(L₁)} = ∅` — that is, none of `L₂`'s operands have their values
determined by `L₁`'s index variable. When this holds, `L₂` computes identical results on
every iteration of `L₁` and can be hoisted entirely outside it.

LICM misses this class of redundancy because it operates at the expression level and cannot
reason about the collective behavior of an entire loop nest as a single reusable unit.

---

## Worked Example: Category 3 Redundancy (O(N³) → O(N²))

Consider the following loop nest, a simplified matrix-vector computation:

```c
// O(N^3)
for (int k = 0; k < N; k++) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            r[k] += a[i][j] * b[j];
        }
    }
}
```

In LER-notation this is:

    ^Rk|0,N|^Si|0,N|^Sj|0,N|a[i,j]*b[j]=r[k]

**relLoops analysis on the inner two loops:**

- `a[i,j]`: index expression uses {i, j} → `relLoops(a) = {i, j}`
- `b[j]`: index expression uses {j} → `relLoops(b) = {j}`
- `relLoops(∑_j a[i,j] * b[j]) = {i, j}`
- `relLoops(∑_i ∑_j a[i,j] * b[j]) = {i, j}`

The index variable of the outer loop is `k`. Since `{i, j} ∩ {k} = ∅`, the entire inner
two-loop nest is loop-invariant with respect to `k` — it computes the same scalar sum on
every iteration of `k`. GLORE hoists it, producing two formulae:

    ^Si|0,N|^Sj|0,N|a[i,j]*b[j]=tmp     // O(N^2), computed once
    ^Rk|0,N|tmp=r[k]                      // O(N),   accumulated N times

**Result:** O(N³) reduced to O(N²). The redundant inner loop nest is separated into its own
formula and executed once before the outer loop, a transformation that is invisible to any
scalar-level analysis.

The Java optimizer (described [below](#java-optimizer-integration)) performs this detection
and emits the rewritten formulae. The `ler-compile` C++ pipeline then lowers each formula
independently to x86.

---

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

## Java Optimizer Integration

The GLORE optimization algorithms themselves are implemented separately in
[optimizer/Glory.java](../optimizer/Glory.java), not inside `ler-compile`. This was a
deliberate pragmatic decision: a working ANTLR-based implementation already existed from
prior compiler optimizations coursework, and reusing it avoided duplicating a substantial
amount of algorithm work during the constrained timeline of this independent study.

### What It Does

The optimizer implements the full GLORE analysis pipeline on a single LER-formula:

1. **Operand folding** — complex sub-expressions are collapsed into synthetic operands, hiding
   internal complexity while exposing loop index dependencies.
2. **Alternating form generation** — converts the expression into a PLUS/TIMES hierarchy,
   enabling free reassociation within each level.
3. **Formula simplification** — breaks a sum-of-products expression into per-term formulae,
   grouping operands by their `relLoops` sets.
4. **Operand abstraction** — replaces each operand's concrete index expression with its
   `relLoops` set — the set of loop IDs relevant to that operand.
5. **Loop encapsulation** — introduces pseudo-bounds to hide variable loop bounds behind
   constants, making all loops appear to have constant trip counts for the purpose of
   redundancy detection.
6. **Minimum union algorithm** — detects category 3 redundancies (loop-invariant reduction
   loops): finds the assignment of each operand to the minimum enclosing loop level consistent
   with its `relLoops` set, and splits formulae to hoist invariant sub-computations.
7. **Operand closure algorithm** — detects category 3 redundancies for regular `for` loops.
8. **Loop decapsulation** — restores concrete loop bounds after redundancy removal.
9. **Partially-loop-invariant handling** — applies category 4 transformations using
   incremental representation: when a later iteration of `L₁` recomputes a strict superset
   of an earlier iteration's inner-loop computation, only the delta is computed.

### Integration with `ler-compile`

The optimizer is a standalone Java process. It reads a `.ler` file, runs the GLORE pipeline,
and writes the transformed LER formulae to stdout. Each output formula is then passed
individually as input to `ler-compile` for lowering to x86. This two-step process is not
yet automated in the build system — see [USAGE.md](./USAGE.md) for the manual workflow.

The optimizer requires the ANTLR 4 runtime on the classpath:

    export CLASSPATH=".:/usr/local/lib/antlr-4.13.1-complete.jar:$CLASSPATH"
    java optimizer.Glory <input.ler>

---

## Known Limitations

- **Hardcoded array sizes.** `ConvertArrayAccToMemref` allocates arrays at compile time with
  fixed upper bounds (1D: 1,000,000 elements; 2D: 1,000×1,000). Arrays with more than two
  dimensions or larger sizes are not supported.

- **Integer-only types.** All values are `i64` throughout the pipeline. Floating-point loop
  bodies are not handled.

- **Conservative loop bounds in `ConvertLoopsToAffineSCF`.** When lowering variable bounds
  like `^Sj|0,i|` (upper bound is a loop index), the pass over-approximates to a constant
  (1,000). This is correct but wasteful and is the messiest part of the current
  implementation.

- **Optimizer is not integrated into the build.** The Java optimizer and `ler-compile` are
  not pipelined — running GLORE optimization requires a manual two-step invocation. A natural
  improvement would be to invoke the optimizer as a subprocess from `LERMain.cpp` and feed
  its output directly to the MLIR codegen stage.

- **No GLORE optimization inside `ler-compile` itself.** The C++ pipeline only lowers
  formulae — it does not detect or remove redundancies. All optimization happens in the Java
  optimizer before any MLIR is generated. If you pass an unoptimized formula directly to
  `ler-compile`, you get unoptimized code.

- **`while` loop support is incomplete.** `ler.While` lowers to `scf.while` with a
  synthetic counter as the loop condition, giving it a fixed iteration count. Loops with
  data-dependent termination are not correctly handled.
