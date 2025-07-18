reticulate::py_require("jax-metal")
reticulate::py_require("numpy")

jax <- reticulate::import("jax")

jax$export$default_export_platform()

invisible(reticulate::py_run_string(
  "

import jax
from jax import export
import jax.numpy as jnp
import numpy as np

# Create a JIT-transformed function
@jax.jit
def plus(x):
  return jnp.add(x,x)

# Create abstract input shapes
inputs = (np.float32(1),)
input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]

# Export the function to StableHLO
exported = export.export(plus, platforms = ['METAL', 'cpu'])(*input_shapes)
stablehlo_add = exported.mlir_module()
"
))

writeLines(
  reticulate::py$stablehlo_add,
  "inst/programs/jax-stablehlo-metal.mlir"
)
