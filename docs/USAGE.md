# `ler-compile` Usage Guide
### See the [README](../README.md) for build instructions.

## Basic Usage

The `ler-compile` executable takes a singular positional argument which is the name of the LER file to be compiled. An example file named [`case10.txt`](../test/case10.txt) contains the LER notation like so:

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

To invoke the compiler, run:

    ./ler-compile case10.txt

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
