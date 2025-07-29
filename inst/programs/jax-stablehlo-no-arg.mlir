module @jit_f attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<3> : tensor<i32> loc(#loc)
    return %c : tensor<i32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)

