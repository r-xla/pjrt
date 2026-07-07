# Tree Size

Counts the number of leaf nodes in a tree. This equals the length of the
flat list produced by
[`flatten()`](https://r-xla.github.io/pjrt/dev/reference/flatten.md) on
the original structure.

## Usage

``` r
tree_size(x)
```

## Arguments

- x:

  (`RTree`)  
  A tree as returned by
  [`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

## Value

`integer(1)`

## See also

[`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md),
[`flatten()`](https://r-xla.github.io/pjrt/dev/reference/flatten.md)

## Examples

``` r
tree_size(build_tree(list(a = 1, b = list(c = 2, d = 3))))
#> [1] 3
```
