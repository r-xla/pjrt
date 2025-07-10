reticulate::py_require("jax")
reticulate::py_require("numpy")

jax <- reticulate::import("jax")

invisible(reticulate::py_run_string(
  "

import jax
from jax import export
import jax.numpy as jnp
import numpy as np

# Create a JIT-transformed function
@jax.jit
def subset(x, i1, i2):
  return x[i1, i2]

# Create abstract input shapes
inputs = (np.float32([[1, 2], [3, 4]]), np.int32(1), np.int32(1))
input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]

# Export the function to StableHLO
exported = export.export(subset)(*input_shapes)
stablehlo_subset = exported.mlir_module()
"
))

writeLines(reticulate::py$stablehlo_subset, "inst/programs/jax-stablehlo-subset.mlir")
