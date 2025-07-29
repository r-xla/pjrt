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
def f():
  return jnp.array(3)


# Export the function to StableHLO
exported = export.export(f)()
stablehlo_f = exported.mlir_module()
"
))

writeLines(
  reticulate::py$stablehlo_f,
  "inst/programs/jax-stablehlo-no-arg.mlir"
)
