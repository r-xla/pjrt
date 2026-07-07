# Per-Leaf Mask From Top-Level Groups

A logical mask over the leaves (in flat order) that is `TRUE` for every
leaf sitting under a top-level child whose name is in `groups`. An empty
`groups` yields an all-`FALSE` mask.

## Usage

``` r
tree_leaf_mask(tree, groups)
```

## Arguments

- tree:

  (`RTree`)  
  A tree whose root is a list node.

- groups:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Top-level child names to mark.

## Value

A `logical` vector of length `tree_size(tree)`.

## See also

[`tree_leaf_groups()`](https://r-xla.github.io/pjrt/dev/reference/tree_leaf_groups.md)

## Examples

``` r
tree_leaf_mask(build_tree(list(a = 1, b = list(2, 3))), "b")
#> [1] FALSE  TRUE  TRUE
```
