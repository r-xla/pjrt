module @jit_i attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xi32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<[1, 2]> : tensor<2xi32> loc(#loc)
    return %c : tensor<2xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)

