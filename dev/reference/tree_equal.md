# Compare Tree Structures

Structural equality of two trees: identical node kinds, child structure,
leaf positions, and names.

## Usage

``` r
tree_equal(a, b)
```

## Arguments

- a, b:

  (`RTree`)  
  Trees to compare, as returned by
  [`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

## Value

`logical(1)`

## See also

[`tree_diff()`](https://r-xla.github.io/pjrt/dev/reference/tree_diff.md)
to locate the first divergence.

## Examples

``` r
tree_equal(build_tree(list(a = 1)), build_tree(list(a = 2)))
#> [1] TRUE
tree_equal(build_tree(list(a = 1)), build_tree(list(b = 1)))
#> [1] FALSE
```
