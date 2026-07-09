# Tree Root Node Kind

The kind of a tree's root node: `"leaf"` for a single leaf, `"list"` for
a list node, or `"null"` for the empty (`NULL`) node.

## Usage

``` r
tree_root_kind(tree)
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
tree_root_kind(build_tree(1))
#> [1] "leaf"
tree_root_kind(build_tree(list(1)))
#> [1] "list"
tree_root_kind(build_tree(NULL))
#> [1] "null"
```
