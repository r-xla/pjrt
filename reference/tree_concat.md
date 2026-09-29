# Concatenate Trees Under a New Root

Builds a new list tree whose children are the given trees, reassigning
leaf indices contiguously across all children (in order).

## Usage

``` r
tree_concat(children, names = NULL)
```

## Arguments

- children:

  (`list` of `RTree`)  
  The child trees.

- names:

  (`NULL` \| [`character()`](https://rdrr.io/r/base/character.html))  
  Optional names for the children.

## Value

A `RTree`.

## Examples

``` r
combined <- tree_concat(
  list(build_tree(list(1, 2)), build_tree(3)),
  names = c("value", "grad")
)
tree_size(combined)
#> [1] 3
```
