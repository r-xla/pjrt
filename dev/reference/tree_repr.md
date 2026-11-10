# Tree Representation

A structural string for a tree: `"*"` for a leaf, `"NULL"` for an empty
node, `"list(...)"` for list nodes, and for an object of a class
registered with
[`register_tree_node()`](https://r-xla.github.io/pjrt/dev/reference/register_tree_node.md)
whatever its `repr` makes of it. A named list is tagged
`"list<named>(...)"`, so that a list whose names are all empty is told
apart from an unnamed one. The string is canonical – trees that are not
[`tree_equal()`](https://r-xla.github.io/pjrt/dev/reference/tree_equal.md)
print differently – as far as the `repr`s of their classes are, and
except for trees built under two registrations of a class, which may
print alike.

## Usage

``` r
tree_repr(tree)
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
tree_repr(build_tree(list(a = 1, b = list(2, NULL))))
#> [1] "list<named>(a = *, b = list(*, NULL))"
```
