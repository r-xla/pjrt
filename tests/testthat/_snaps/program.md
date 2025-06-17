# load a test program

    Code
      print(program)
    Output
      PJRTProgram(format=hlo, code_size=158)
      name: "x*x.3"
      entry_computation_name: "x*x.3"
      computations {
        name: "x*x.3"
        instructions {
      ... 

# can load MLIR program

    Code
      print(program)
    Output
      PJRTProgram(format=mlir, code_size=587)
      #loc1 = loc("x")
      module @jit_plus attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
        func.func public @main(%arg0: tensor<f32> loc("x")) -> (tensor<f32> {jax.result_info = ""}) {
          %0 = stablehlo.add %arg0, %arg0 : tensor<f32> loc(#loc7)
          return %0 : tensor<f32> loc(#loc)
      ... 

