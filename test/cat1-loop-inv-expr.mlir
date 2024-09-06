#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (100)>
"func.func"() <{function_type = () -> (), sym_name = "cat1"}> ({
  %0 = "arith.constant"() <{value = 20 : i64}> : () -> i64
  %1 = "arith.constant"() <{value = 30 : i64}> : () -> i64
  %2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<100xi64>
  %3 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<100xi64>
  "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
  ^bb0(%arg0: index):
    %4 = "arith.muli"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %5 = "memref.load"(%3, %arg0) <{nontemporal = false}> : (memref<100xi64>, index) -> i64
    %6 = "arith.addi"(%4, %5) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    "memref.store"(%6, %2, %arg0) <{nontemporal = false}> : (i64, memref<100xi64>, index) -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
}) : () -> ()
