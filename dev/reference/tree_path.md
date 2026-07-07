# Tree Path

Returns the human-readable path for a single leaf identified by its flat
index (e.g. `"a$b"` or `"[[2]]"`). Only descends into the branch
containing the target leaf, making it efficient for error reporting.

## Usage

``` r
tree_path(tree, i)
```

## Arguments

- tree:

  (`RTree`)  
  A tree as returned by
  [`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

- i:

  (`integer(1)`)  
  The flat index of the leaf.

## Value

`character(1)` (`""` for a leaf at the root).

## See also

[`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md),
[`flatten()`](https://r-xla.github.io/pjrt/dev/reference/flatten.md)

## Examples

``` r
tree_path(build_tree(list(a = 1, b = list(c = 2))), 2L)
#> [1] "b$c"
```
