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
def f():
  return jnp.array(3)

# Export the function to StableHLO
exported = export.export(f)()
stablehlo_f = exported.mlir_module()

# Create a JIT-transformed function that returns two constants
@jax.jit
def g():
  return jnp.array(3), jnp.array(7)

exported_g = export.export(g)()
stablehlo_two_constants = exported_g.mlir_module()

@jax.jit
def i():
  return jnp.array([[1, 2], [3, 4]])

exported_i = export.export(i)()
stablehlo_tensor_constant = exported_i.mlir_module()

@jax.jit
def j():
  return jnp.array(1.0)

exported_j = export.export(j)()
stablehlo_float_constant = exported_j.mlir_module()

@jax.jit
def k():
  return jnp.array([[1.0, 2.0]])

exported_k = export.export(k)()
stablehlo_float_tensor_constant = exported_k.mlir_module()
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

writeLines(
  reticulate::py$stablehlo_tensor_constant,
  "inst/programs/jax-stablehlo-tensor-constant.mlir"
)
writeLines(
  reticulate::py$stablehlo_float_constant,
  "inst/programs/jax-stablehlo-float-constant.mlir"
)
writeLines(
  reticulate::py$stablehlo_float_tensor_constant,
  "inst/programs/jax-stablehlo-float-tensor-constant.mlir"
)
