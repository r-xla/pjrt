# Difference Between Trees

Walks two trees in parallel and returns the first path/subtree pair
where they diverge, or `NULL` if they are structurally identical. The
returned `prefix` follows
[`tree_path()`](https://r-xla.github.io/pjrt/dev/reference/tree_path.md)
syntax; `a` and `b` are the
[`tree_repr()`](https://r-xla.github.io/pjrt/dev/reference/tree_repr.md)
strings of the diverging subtrees.

## Usage

``` r
tree_diff(a, b)
```

## Arguments

- a, b:

  (`RTree`)  
  Trees to compare, as returned by
  [`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

## Value

`NULL` if `a` and `b` are structurally identical, otherwise a list with
elements `prefix`, `a`, and `b`.

## See also

[`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md),
[`tree_path()`](https://r-xla.github.io/pjrt/dev/reference/tree_path.md),
[`tree_equal()`](https://r-xla.github.io/pjrt/dev/reference/tree_equal.md)

## Examples

``` r
tree_diff(build_tree(list(a = 1)), build_tree(list(a = list(1, 2))))
#> $prefix
#> [1] "a"
#> 
#> $a
#> [1] "*"
#> 
#> $b
#> [1] "list(*, *)"
#> 
```
