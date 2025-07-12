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
stablehlo_subset_2d = exported.mlir_module()
"
))

writeLines(
  reticulate::py$stablehlo_subset_2d,
  "inst/programs/jax-stablehlo-subset-2d.mlir"
)

invisible(reticulate::py_run_string(
  "
import jax
from jax import export
import jax.numpy as jnp
import numpy as np

# Create a JIT-transformed function
@jax.jit
def subset(x, i1, i2, i3):
  return x[i1, i2, i3]

# Create abstract input shapes
inputs = (np.float32(np.arange(1, 25).reshape(2, 3, 4)), np.int32(1), np.int32(1), np.int32(1))
input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]

# Export the function to StableHLO
exported = export.export(subset)(*input_shapes)
stablehlo_subset_3d = exported.mlir_module()
"
))

writeLines(
  reticulate::py$stablehlo_subset_3d,
  "inst/programs/jax-stablehlo-subset-3d.mlir"
)
