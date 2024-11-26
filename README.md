# LER-IR Compiler

LER-IR is an frontenc compiler and MLIR dialect for compiling LER notation into executables. It is based of the paper: [‘GLORE: Generalized Loop Redundancy Elimination upon LER-Notation’](https://research.csc.ncsu.edu/picture/publications/papers/oopsla17.pdf) (Shen, Ding 2017).

## Getting Started
To get started with this project, ensure you have the following installed on your system:

- [CMake >= v3.22](https://cmake.org/download/)
- [LLVM/MLIR Libraries >= v18.1.6](https://github.com/llvm/llvm-project)
  - Instructions for build LLVM with MLIR can be found [here](https://mlir.llvm.org/getting_started/).
- [Ninja](https://ninja-build.org/) build system, this is _optional_ but recommended.

### Java Optimizer Integration
The LER optimizer in the [optimizer](./optimizer) directory requires the Java ANTLR parser library to be compiled:

    cd /usr/local/lib
    sudo curl -O https://www.antlr.org/download/antlr-4.13.1-complete.jar
    export CLASSPATH=".:/usr/local/lib/antlr-4.13.1-complete.jar:$CLASSPATH"

### GNU/UNIX Make Build
```shell
git clone https://github.com/bradenhelmer/LER-IR
mkdir build
cmake -S . -B build
cmake --build build
```
### Ninja Build
```shell
git clone https://github.com/bradenhelmer/LER-IR
mkdir build
cmake -S . -G Ninja -B build
cmake --build build
```

An executable named `ler-compile` will be in the `build` directory.

## [USAGE](./docs/USAGE.md)

## [INTERNALS](./docs/INTERNALS.md)

## [LER DIALECT & OPERATIONS](./docs/LERDialect.md) (MLIR Auto Generated)

## [LER PASSES](./docs/LERPasses.md) (MLIR Auto Generated)
