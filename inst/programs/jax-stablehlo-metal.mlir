#loc = loc(unknown)
#loc1 = loc("x")
module @jit_plus attributes {jax.uses_shape_polymorphism = true, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i32> {jax.global_constant = "_platform_index"} loc(unknown), %arg1: tensor<f32> loc("x")) -> (tensor<f32> {jax.result_info = "result"}) {
    %0 = stablehlo.add %arg1, %arg1 : tensor<f32> loc(#loc7)
    return %0 : tensor<f32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("<string>":11:9 to :21)
#loc3 = loc("<string>":18:11 to :75)
#loc4 = loc("plus"(#loc2))
#loc5 = loc("<module>"(#loc3))
#loc6 = loc(callsite(#loc4 at #loc5))
#loc7 = loc("jit(plus)/jit(main)/add"(#loc6))

