# Top-Level Child Kinds

The node kind (`"leaf"`, `"list"`, or `"null"`) of each top-level child
of a list tree.

## Usage

``` r
tree_child_kinds(tree)
```

## Arguments

- tree:

  (`RTree`)  
  A tree whose root is a list node.

## Value

A `character` vector, one element per top-level child.

## Examples

``` r
tree_child_kinds(build_tree(list(1, list(2), NULL)))
#> [1] "leaf" "list" "null"
```
