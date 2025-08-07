func.func public @main() -> (tensor<f32> {jax.result_info = "result"}) {
  %cst = stablehlo.constant dense<3.14000000e+00> : tensor<f32> loc(#loc)
  return %cst : tensor<f32> loc(#loc)
} loc(#loc)
