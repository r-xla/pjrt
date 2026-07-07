# Flatten a Function

Flattens a function taking in possibly nested `RTree` objects so it
takes in flat inputs and return flat outputs (along the returned tree
structure).

## Usage

``` r
flatten_fun(f, in_tree)
```

## Arguments

- f:

  (`function`)  
  Function to wrap.

- in_tree:

  (`RTree`)  
  Tree describing the (possibly nested) argument list of `f`, i.e.
  `build_tree(list(arg1, arg2, ...))`. The flat inputs to the wrapper
  follow this structure.

## Value

A `FlattenedFunction`.

## See also

[`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md),
[`flatten()`](https://r-xla.github.io/pjrt/dev/reference/flatten.md),
[`unflatten()`](https://r-xla.github.io/pjrt/dev/reference/unflatten.md)

## Examples

``` r
f <- function(x) list(sum = x$a + x$b)
ff <- flatten_fun(f, build_tree(list(list(a = 1, b = 2))))
ff(3, 4)
#> [[1]]
#> list(sum = *)
#> 
#> [[2]]
#> [[2]][[1]]
#> [1] 7
#> 
#> 
```
