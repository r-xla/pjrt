# Copy Buffer to Device

Copy a
[`PJRTBuffer`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.md) to
a different device. Returns a new buffer on the target device; the
original buffer is unchanged.

If the buffer already lives in the requested device, no copy is
performed.

When the target device belongs to a different client (e.g. copying from
CPU to CUDA), the transfer is performed via a host roundtrip.

## Usage

``` r
copy_buffer(buffer, device)
```

## Arguments

- buffer:

  ([`PJRTBuffer`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.md))  
  A PJRT buffer object.

- device:

  (`PJRTDevice` \| `character(1)`)  
  The target device. A `PJRTDevice` object or a device specification
  (e.g., `"cpu:0"`, `"cpu:1"`, `"cuda:0"`).

## Value

A new `PJRTBuffer` on the target device.

## Examples

``` r
if (FALSE) { # plugins_downloaded(c("cpu", "cuda"))
buf <- pjrt_buffer(c(1, 2, 3), device = "cpu")
buf2 <- copy_buffer(buf, "cuda")
device(buf2)
}
```
