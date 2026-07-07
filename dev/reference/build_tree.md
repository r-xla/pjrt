# Build Tree

Captures the nesting structure of an object as an opaque `RTree`. Each
leaf in the input corresponds to its position in the flat list produced
by [`flatten()`](https://r-xla.github.io/pjrt/dev/reference/flatten.md).
The resulting tree can be passed to
[`unflatten()`](https://r-xla.github.io/pjrt/dev/reference/unflatten.md)
to reconstruct the original structure from a flat list.

`NULL` is an empty node: it holds no leaf and contributes nothing to the
flat list, but is preserved in the tree (so `list(x, NULL)` and
`list(NULL, x)` have distinct trees).

The `RTree` and its operations are pjrt's R analog of [JAX's
pytree](https://docs.jax.dev/en/latest/pytrees.html), which is where the
idea comes from.

## Usage

``` r
build_tree(x)
```

## Arguments

- x:

  (any)  
  Object whose structure to capture. Bare lists are recursed into;
  everything else is a leaf.

## Value

A `RTree`.

## See also

[`flatten()`](https://r-xla.github.io/pjrt/dev/reference/flatten.md),
[`unflatten()`](https://r-xla.github.io/pjrt/dev/reference/unflatten.md),
[`tree_size()`](https://r-xla.github.io/pjrt/dev/reference/tree_size.md)

## Examples

``` r
x <- list(a = 1, b = list(c = 2, d = 3))
tree <- build_tree(x)
tree_size(tree)
#> [1] 3

unflatten(tree, flatten(x))
#> $a
#> [1] 1
#> 
#> $b
#> $b$c
#> [1] 2
#> 
#> $b$d
#> [1] 3
#> 
#> 
```
