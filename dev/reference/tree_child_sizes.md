# Top-Level Child Sizes

The number of leaves under each top-level child of a list tree.

## Usage

``` r
tree_child_sizes(tree)
```

## Arguments

- tree:

  (`RTree`)  
  A tree whose root is a list node.

## Value

An `integer` vector, one element per top-level child.

## Examples

``` r
tree_child_sizes(build_tree(list(a = 1, b = list(2, 3), c = NULL)))
#> [1] 1 2 0
```
