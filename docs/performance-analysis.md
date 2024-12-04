# Performance Analysis of Test Cases

These are preliminary results of the test cases lowered into executables without any GLORE optimization described in the paper.

## Machine Specifications
- **Operating System**: x86_64 GNU/Linux (Pop-OS)
- **CPU**: AMD Ryzen 9 7900X 12-Core Processor @ 5.7GHZ
- **Caches** (sum of all):
  - L1d: 384 KiB (12 instances)
  - L1i: 384 KiB (12 instances)
  - L2: 12 MiB (12 instances)
  - L3: 64 MiB (2 instances)
- **Memory**: 64GB

## Compiler Specifcations
- **llc**: v18.1.6 with `-O3` flags.
- **clang-as**: v18.1.6 with `-O3` flags.

## Analysis Table

These results were recorded using the `perf` tool with core pinning:

    taskset -c 0 perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ./<TESTCASE>

| Test Case                        | Expected Iterations | Runtime (s) | Cycles         | Instructions    | IPC  | Branch Miss/Hit Ratio | Cache Miss/Hit Ratio |
| -------------------------------- | ------------------- | ----------- | -------------- | --------------- | ---- | --------------------- | -------------------- |
| [case](../test/case.ler)         | 1 trillion          | 674         | 3.695 trillion | 8.976 trillion  | 2.43 | 0.001                 | 0.000000488          |
| [case2](../test/case2.ler)       | 1 billion           | 0.534       | 2.654 billion  | 6 billion       | 2.26 | 0.00000822            | 0.00000843           |
| [case4.1](../test/case4.1.ler)   | 1 trillion          | 244         | 1.34 trillion  | 6.011 trillion  | 4.48 | 0.001                 | 0.000000482          |
| [case4.2](../test/case4.2.ler)   | 1 trillion          | 235         | 1.292 trillion | 5.011 trillion  | 3.88 | 0.001                 | 0.00000206           |
| [case4.3](../test/case4.3.ler)   | 1 trillion          | 247         | 1.362 trillion | 5.000 trillion  | 3.67 | 0.00000143            | 0.016                |
| [case5](../test/case5.ler)       | 1 million           | 0.000000091 | 2.071 million  | 5.161 million   | 2.49 | 0.005                 | 0.0258               |
| [case7](../test/case7.ler)       | 1 billion           | 0.531       | 2.637 billion  | 6.000 billion   | 2.28 | 0.00000823            | 0.00000829           |
| [case8](../test/case8-fused.ler) | 1 trillion          | 184         | 1.005 trillion | 5.000 trillion  | 4.97 | 0.00000136            | 0.00000219           |
| [case9](../test/case9.ler)       | 1 trillion          | 286         | 1.565 trillion | 6.000 trillion  | 3.83 | 0.00000157            | 0.000002             |
| [case10](../test/case10.ler)     | 1 billion           | 0.62        | 3.092 billion  | 7.003 billion   | 2.26 | 0.0001                | 0.00000788           |
