reticulate::py_require("jax")
reticulate::py_require("numpy")

jax <- reticulate::import("jax")

invisible(reticulate::py_run_string(
  "

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
from functools import partial

@partial(jax.jit, donate_argnames=['x'])

@jax.jit
def update_param(x, grad):
  return x - grad

# Create abstract input shapes for 1000x1000 array
input_shapes = [jax.ShapeDtypeStruct((1000000,), np.float32), jax.ShapeDtypeStruct((1000000,), np.float32)]

# Export the function to StableHLO
exported = export.export(update_param)(*input_shapes)
stablehlo_update_param = exported.mlir_module()
"
))

writeLines(reticulate::py$stablehlo_update_param, "inst/programs/jax-stablehlo-update-param.mlir")
