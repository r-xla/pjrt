#loc1 = loc("x")
#loc2 = loc("grad")
module @jit_update_param attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1000000xf32> {tf.aliasing_output = 0 : i32} loc("x"), %arg1: tensor<1000000xf32> loc("grad")) -> (tensor<1000000xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<1000000xf32> loc(#loc8)
    return %0 : tensor<1000000xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("<string>":11:9 to :17)
#loc4 = loc("<string>":17:11 to :53)
#loc5 = loc("update_param"(#loc3))
#loc6 = loc("<module>"(#loc4))
#loc7 = loc(callsite(#loc5 at #loc6))
#loc8 = loc("jit(update_param)/jit(main)/sub"(#loc7))

