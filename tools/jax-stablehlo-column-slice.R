reticulate::py_require("jax")
reticulate::py_require("numpy")

jax <- reticulate::import("jax")

# Function 1: Slice column and drop other dimensions (returns 1D tensor)
invisible(reticulate::py_run_string(
  "
import jax
from jax import export
import jax.numpy as jnp
import numpy as np

# Create a JIT-transformed function that slices a column from 3D tensor
# and drops other dimensions (returns 1D tensor)
@jax.jit
def slice_column_drop(x, col_idx):
  return x[:, col_idx]

# Create abstract input shapes
inputs = (np.float32(np.arange(1, 13).reshape(3, 4)), np.int32(1))
input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]

# Export the function to StableHLO
exported = export.export(slice_column_drop)(*input_shapes)
stablehlo_slice_column_drop = exported.mlir_module()
"
))

writeLines(
  reticulate::py$stablehlo_slice_column_drop,
  "inst/programs/jax-stablehlo-slice-column-drop.mlir"
)

# Function 2: Slice column but keep other dimensions (returns 2D tensor)
invisible(reticulate::py_run_string(
  "
import jax
from jax import export
import jax.numpy as jnp
import numpy as np

# Create a JIT-transformed function that slices a column from 3D tensor
# but keeps other dimensions (returns 2D tensor)
@jax.jit
def slice_column_keep(x, col_idx):
  return x[:, [col_idx]]

# Create abstract input shapes
inputs = (np.float32(np.arange(1, 13).reshape(3, 4)), np.int32(1))
input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]

# Export the function to StableHLO
exported = export.export(slice_column_keep)(*input_shapes)
stablehlo_slice_column_keep = exported.mlir_module()
"
))

writeLines(
  reticulate::py$stablehlo_slice_column_keep,
  "inst/programs/jax-stablehlo-slice-column-keep.mlir"
)
