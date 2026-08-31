module @jit_mlp attributes {jax.uses_shape_polymorphism = true, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4x2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<?x3xf32>) -> (tensor<?x2xf32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.get_dimension_size %arg4, dim = 0 : (tensor<?x3xf32>) -> tensor<i32>
    %1 = stablehlo.compare  GE, %0, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%1, %0) {api_version = 2 : i32, error_message = "Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable 'batch'. Using the following polymorphic shapes specifications: args[4].shape = (batch, 3). Obtained dimension variables: 'batch' = {0} from specification 'batch' for dimension args[4].shape[0] (= {0}), . Please see https://docs.jax.dev/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
    %2 = call @_wrapped_jax_export_main(%0, %arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<i32>, tensor<3x4xf32>, tensor<4xf32>, tensor<4x2xf32>, tensor<2xf32>, tensor<?x3xf32>) -> tensor<?x2xf32>
    return %2 : tensor<?x2xf32>
  }
  func.func private @_wrapped_jax_export_main(%arg0: tensor<i32> {jax.global_constant = "batch"}, %arg1: tensor<3x4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4x2xf32>, %arg4: tensor<2xf32>, %arg5: tensor<?x3xf32>) -> (tensor<?x2xf32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<2> : tensor<1xi32>
    %c_0 = stablehlo.constant dense<4> : tensor<1xi32>
    %0 = stablehlo.dot_general %arg5, %arg1, contracting_dims = [1] x [0] : (tensor<?x3xf32>, tensor<3x4xf32>) -> tensor<?x4xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.concatenate %2, %c_0, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = stablehlo.dynamic_broadcast_in_dim %1, %3, dims = [0, 1] : (tensor<1x4xf32>, tensor<2xi32>) -> tensor<?x4xf32>
    %5 = stablehlo.add %0, %4 : tensor<?x4xf32>
    %6 = stablehlo.tanh %5 : tensor<?x4xf32>
    %7 = stablehlo.dot_general %6, %arg3, contracting_dims = [1] x [0] : (tensor<?x4xf32>, tensor<4x2xf32>) -> tensor<?x2xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %9 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.concatenate %9, %c, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %11 = stablehlo.dynamic_broadcast_in_dim %8, %10, dims = [0, 1] : (tensor<1x2xf32>, tensor<2xi32>) -> tensor<?x2xf32>
    %12 = stablehlo.add %7, %11 : tensor<?x2xf32>
    return %12 : tensor<?x2xf32>
  }
}

