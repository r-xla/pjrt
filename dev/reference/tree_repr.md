# Canonical Tree Representation

A canonical structural string for a tree: `"*"` for a leaf, `"NULL"` for
an empty node, and `"list(...)"` for list nodes.

## Usage

``` r
tree_repr(tree)
```

## Arguments

- tree:

  (`RTree`)  
  A tree as returned by
  [`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

## Value

`character(1)`

## Examples

``` r
tree_repr(build_tree(list(a = 1, b = list(2, NULL))))
#> [1] "list<named>(a = *, b = list(*, NULL))"
```
