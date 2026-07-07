# Per-Leaf Top-Level Group Names

For each leaf (in flat order), the name of the top-level child it sits
under (`""` for unnamed children).

## Usage

``` r
tree_leaf_groups(x)
```

## Arguments

- x:

  (`RTree`)  
  A tree whose root is a list node.

## Value

A `character` vector of length `tree_size(x)`.

## Examples

``` r
tree_leaf_groups(build_tree(list(a = 1, b = list(2, 3))))
#> [1] "a" "b" "b"
```
