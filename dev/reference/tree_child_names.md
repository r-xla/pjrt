# Top-Level Child Names

The names of a list tree's top-level children (with `""` for unnamed
slots), or `NULL` if the root is not a list node or carries no names.

## Usage

``` r
tree_child_names(tree)
```

## Arguments

- tree:

  (`RTree`)  
  A tree as returned by
  [`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

## Value

A `character` vector of child names, or `NULL` when the root is not a
list node or is an *unnamed* list. An empty but named list
(`structure(list(), names = character(0))`) returns `character(0)`, not
`NULL`: the tree preserves the named/unnamed distinction (see
[`tree_equal()`](https://r-xla.github.io/pjrt/dev/reference/tree_equal.md)).

## Examples

``` r
tree_child_names(build_tree(list(a = 1, b = 2)))
#> [1] "a" "b"
tree_child_names(build_tree(list(1, 2)))
#> NULL
```
