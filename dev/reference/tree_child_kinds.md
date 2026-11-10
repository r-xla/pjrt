# Top-Level Child Kinds

The node kind (`"leaf"`, `"list"`, `"object"`, or `"null"`; see
[`tree_root_kind()`](https://r-xla.github.io/pjrt/dev/reference/tree_root_kind.md))
of each top-level child of a list or object tree.

## Usage

``` r
tree_child_kinds(tree)
```

## Arguments

- tree:

  (`RTree`)  
  A tree whose root is a list or object node.

## Value

A `character` vector, one element per top-level child.

## Examples

``` r
tree_child_kinds(build_tree(list(1, list(2), NULL)))
#> [1] "leaf" "list" "null"
```
