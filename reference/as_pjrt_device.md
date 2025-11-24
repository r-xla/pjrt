# Convert to PJRT Device

Convert a platform name or device to a PJRT device object.

## Usage

``` r
as_pjrt_device(x)
```

## Arguments

- x:

  (`PJRTDevice` \| `character(1)` \| `NULL`)  
  Either a PJRT device object, a platform name (e.g., "cpu", "cuda",
  "metal"), a device specification with index (e.g., "cpu:0", "cuda:1"
  for 0-based indexing), or NULL (defaults to first CPU device).

## Value

`PJRTDevice`
