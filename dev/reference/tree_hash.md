# Structural Tree Hash

A hash of a tree's structure – node kinds, child structure, leaf
positions, names, and the class and aux data of each object of a class
registered with
[`register_tree_node()`](https://r-xla.github.io/pjrt/dev/reference/register_tree_node.md)
– consistent with
[`tree_equal()`](https://r-xla.github.io/pjrt/dev/reference/tree_equal.md):
trees that are
[`tree_equal()`](https://r-xla.github.io/pjrt/dev/reference/tree_equal.md)
hash to the same value. Intended as a cache key for tree-structured
dispatch within a session: a class folds in by the address of its
registration, so the hash of a tree with registered objects differs
between sessions.

## Usage

``` r
tree_hash(tree)
```

## Arguments

- tree:

  (`RTree`)  
  A tree as returned by
  [`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

## Value

`character(1)`, the structural hash.

## See also

[`tree_equal()`](https://r-xla.github.io/pjrt/dev/reference/tree_equal.md),
[`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md)

## Examples

``` r
tree_hash(build_tree(list(a = 1))) == tree_hash(build_tree(list(a = 2)))
#> [1] TRUE
tree_hash(build_tree(list(a = 1))) == tree_hash(build_tree(list(b = 1)))
#> [1] FALSE
```
