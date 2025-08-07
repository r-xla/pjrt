module @jit_j attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<f32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc)
    return %cst : tensor<f32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)

