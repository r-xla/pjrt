#loc1 = loc("x")
#loc2 = loc("grad")
#loc3 = loc("<string>":19:11 to :53)
#loc5 = loc("<module>"(#loc3))
#loc7 = loc("jit(update_param)/jit(main)/pjit"(#loc5))
module @jit_update_param attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1000000xf32> {tf.aliasing_output = 0 : i32} loc("x"), %arg1: tensor<1000000xf32> loc("grad")) -> (tensor<1000000xf32> {jax.result_info = "result"}) {
    %0 = call @update_param(%arg0, %arg1) : (tensor<1000000xf32>, tensor<1000000xf32>) -> tensor<1000000xf32> loc(#loc7)
    return %0 : tensor<1000000xf32> loc(#loc)
  } loc(#loc)
  func.func private @update_param(%arg0: tensor<1000000xf32> loc("jit(update_param)/jit(main)/pjit"(#loc5)), %arg1: tensor<1000000xf32> loc("jit(update_param)/jit(main)/pjit"(#loc5))) -> tensor<1000000xf32> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<1000000xf32> loc(#loc9)
    return %0 : tensor<1000000xf32> loc(#loc7)
  } loc(#loc7)
} loc(#loc)
#loc = loc(unknown)
#loc4 = loc("<string>":13:9 to :17)
#loc6 = loc("update_param"(#loc4))
#loc8 = loc(callsite(#loc6 at #loc5))
#loc9 = loc("jit(update_param)/jit(main)/jit(update_param)/sub"(#loc8))

