reticulate::py_require("jax")
reticulate::py_require("numpy")

jax <- reticulate::import("jax")

invisible(reticulate::py_run_string(
  "

import jax
from jax import export
import jax.numpy as jnp
import numpy as np

# Create a JIT-transformed function that returns one constant
@jax.jit
def f(x):
  return jnp.array(3)

# Create abstract input shapes
inputs = (np.float32(1),)
input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]

# Export the function to StableHLO
exported = export.export(f)(*input_shapes)
stablehlo_f = exported.mlir_module()

# Create a JIT-transformed function that returns two constants
@jax.jit
def g(x):
  return jnp.array(3), jnp.array(7)

# Export the second function to StableHLO
exported_g = export.export(g)(*input_shapes)
stablehlo_two_constants = exported_g.mlir_module()
"
))

writeLines(
  reticulate::py$stablehlo_f,
  "inst/programs/jax-stablehlo-no-arg.mlir"
)
writeLines(
  reticulate::py$stablehlo_two_constants,
  "inst/programs/jax-stablehlo-two-constants.mlir"
)
