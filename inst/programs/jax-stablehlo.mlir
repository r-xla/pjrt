#loc1 = loc("x")
module @jit_plus attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f32> loc("x")) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<f32> loc(#loc7)
    return %0 : tensor<f32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("<string>":11:0)
#loc3 = loc("<string>":18:0)
#loc4 = loc("plus"(#loc2))
#loc5 = loc("<module>"(#loc3))
#loc6 = loc(callsite(#loc4 at #loc5))
#loc7 = loc("jit(plus)/jit(main)/add"(#loc6))
