module @jit_k attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x2xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32> loc(#loc)
    return %cst : tensor<1x2xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)

