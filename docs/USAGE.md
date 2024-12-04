# `ler-compile` Usage Guide
### See the [README](../README.md) for build instructions.

## Basic Usage

The `ler-compile` executable takes a singular positional argument which is the name of the LER file to be compiled. An example file named [`case10.ler`](../test/case10.ler) contains the LER notation like so:

    Γi∫1,M∫Σk∫0,i∫Σj∫0,i∫x[i,j] * y[j,k] = r[i]

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

1. First convert the LER to the version accepted by the compiler with [this python script](../scripts/notation_converter.py):

        python3 notation_converter.py case10.ler case10.ler_conv

2. Then invoke the compiler on the converted file like so:

        ./ler-compile case10.ler_conv

Simply running the compiler like this will **nots** produce any output. There are many command line options that specify the behavior of the compiler. 

| Option                  | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `--exe`                 | Produces an executable.                        |
| `--output-asm`          | Prints intermediary assembly language to file. |
| `--output-ler-mlir`     | Prints LER dialect MLIR to file.               |
| `--output-lowered-mlir` | Prints lowered LLVM dialect to file.           |
| `--output-llvm-ir`      | Prints intermediary LLVM IR to file.           |
| `--print-ast`           | Prints LER AST to STDOUT.                      |
| `--print-ler-mlir`      | Prints LER dialect MLIR to STDOUT.             |
| `--print-lowered-mlir`  | Prints lowered LLVM dialect to STDOUT.         |

## Integrated Java Optimizer Use (Not Stable)
Additionally, a script is [provided](../scripts/ler-full) to invoked the Java LER optimizer, converter, and compiler. First ensure that the ler-compile binary has been built and run the script in the build directory:

    cd build
    ./ler-full case10.ler
**Warning** The executable lowering stages are currently not working with multilined LER statements.
