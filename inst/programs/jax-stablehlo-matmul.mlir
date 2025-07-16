#loc1 = loc("x")
#loc2 = loc("y")
module @jit_matmul attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1000000xf32> loc("x"), %arg1: tensor<1000000xf32> loc("y")) -> (tensor<f32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<1000000xf32>, tensor<1000000xf32>) -> tensor<f32> loc(#loc8)
    return %0 : tensor<f32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("<string>":10:9 to :14)
#loc4 = loc("<string>":16:11 to :47)
#loc5 = loc("matmul"(#loc3))
#loc6 = loc("<module>"(#loc4))
#loc7 = loc(callsite(#loc5 at #loc6))
#loc8 = loc("jit(matmul)/jit(main)/dot_general"(#loc7))

