# LER-IR Compiler

LER-IR is an frontenc compiler and MLIR dialect for optimizing redudancy across loops. It is based of the paper: [‘GLORE: Generalized Loop Redundancy Elimination upon LER-Notation’](https://research.csc.ncsu.edu/picture/publications/papers/oopsla17.pdf) (Shen, Ding 2017).

## Getting Started
To get started with this project, ensure you have the following installed on your system:

- [CMake >= v3.22](https://cmake.org/download/)
- [LLVM/MLIR Libraries >= v18.1.6](https://github.com/llvm/llvm-project)
  - Instructions for build LLVM with MLIR can be found [here](https://mlir.llvm.org/getting_started/).
- [Ninja](https://ninja-build.org/) build system, this is _optional_ but recommended.

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

An executable named `ler-opt` will be in the `build` directory.

## [USAGE]()

## [INTERNALS]()

## [LER DIALECT & OPERATIONS](./docs/LERDialect.md)

## [LER PASSES](./docs/LERPasses.md)
