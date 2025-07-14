reticulate::py_require("jax")
reticulate::py_require("numpy")

jax <- reticulate::import("jax")

invisible(reticulate::py_run_string(
  "

import jax
from jax import export
import jax.numpy as jnp
import numpy as np

@jax.jit
def modify_element_00(x, grad):
  return x - grad

# Create abstract input shapes for 1000x1000 array
input_shapes = [jax.ShapeDtypeStruct((1000000,), np.float32), jax.ShapeDtypeStruct((1000000,), np.float32)]

# Export the function to StableHLO
exported = export.export(modify_element_00)(*input_shapes)
stablehlo_modify_element = exported.mlir_module()
"
))

writeLines(reticulate::py$stablehlo_modify_element, "inst/programs/jax-stablehlo-modify-element.mlir")
