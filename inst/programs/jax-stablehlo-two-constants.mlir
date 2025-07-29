module @jit_g attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i32> {jax.result_info = "result[0]"}, tensor<i32> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<3> : tensor<i32> loc(#loc)
    %c_0 = stablehlo.constant dense<7> : tensor<i32> loc(#loc)
    return %c, %c_0 : tensor<i32>, tensor<i32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)

