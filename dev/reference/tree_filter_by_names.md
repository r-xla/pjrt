# Filter a Tree by Top-Level Names

Keeps only the top-level children whose names match `names` (in tree
order), reindexing the remaining leaves so they map to contiguous
positions in a flat list.

## Usage

``` r
tree_filter_by_names(tree, names)
```

## Arguments

- tree:

  (`RTree`)  
  A named list tree as returned by
  [`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

- names:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of children to keep.

## Value

A `RTree` containing only the selected children.

## See also

[`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md),
[`unflatten()`](https://r-xla.github.io/pjrt/dev/reference/unflatten.md)

## Examples

``` r
x <- list(a = 1, b = 2, c = 3)
sub <- tree_filter_by_names(build_tree(x), c("a", "c"))
tree_size(sub)
#> [1] 2
unflatten(sub, x[c("a", "c")])
#> $a
#> [1] 1
#> 
#> $c
#> [1] 3
#> 
```
