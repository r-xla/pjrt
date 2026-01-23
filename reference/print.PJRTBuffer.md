# Print a PJRT Buffer

Print a [`PJRTBuffer`](pjrt_buffer.md).

## Usage

``` r
# S3 method for class 'PJRTBuffer'
print(
  x,
  max_rows = getOption("pjrt.print_max_rows", 30L),
  max_width = getOption("pjrt.print_max_width", 85L),
  max_rows_slice = getOption("pjrt.print_max_rows_slice", max_rows),
  header = TRUE,
  footer = NULL,
  ...
)
```

## Arguments

- x:

  (`PJRTBuffer`)  
  The buffer.

- max_rows:

  (`integer(1)`)  
  The maximum number of rows to print, excluding header and footer.

- max_width:

  (`integer(1)`)  
  The maximum width (in characters) of the printed buffer. Set to
  negative values for no limit. Note that for very small values, the
  actual printed width might be slightly smaller as at least one column
  will be printed. Also, this limit only affects the printed rows
  containing the actual data, other rows might exceed the width.

- max_rows_slice:

  (`integer(1)`)  
  The maximum number of rows to print for each slice.

- header:

  (`logical(1)`)  
  Whether to print the header.

- footer:

  (`NULL` or `character(1)`)  
  The footer line to print. If `NULL` (default), prints the standard
  `[ <PLATFORM><TYPE>{<SHAPE>} ]` summary. Use `""` to suppress.

- ...:

  Additional arguments (unused).
