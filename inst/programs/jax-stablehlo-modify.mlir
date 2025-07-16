#loc1 = loc("x")
#loc2 = loc("<string>":10:6 to :21)
#loc3 = loc("<string>":17:11 to :47)
#loc4 = loc("modify"(#loc2))
#loc5 = loc("<module>"(#loc3))
#loc6 = loc(callsite(#loc4 at #loc5))
#loc8 = loc("jit(modify)/jit(main)/scatter"(#loc6))
module @jit_modify attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1000000xf32> loc("x")) -> (tensor<1000000xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<4.200000e+01> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc7)
    %1 = "stablehlo.scatter"(%arg0, %0, %cst) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<f32> loc("jit(modify)/jit(main)/scatter"(#loc6)), %arg2: tensor<f32> loc("jit(modify)/jit(main)/scatter"(#loc6))):
      stablehlo.return %arg2 : tensor<f32> loc(#loc8)
    }) : (tensor<1000000xf32>, tensor<1xi32>, tensor<f32>) -> tensor<1000000xf32> loc(#loc8)
    return %1 : tensor<1000000xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc7 = loc("jit(modify)/jit(main)/broadcast_in_dim"(#loc6))

