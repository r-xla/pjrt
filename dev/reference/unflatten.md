# Unflatten

Reconstructs a nested structure from a flat list by using a tree
previously created with
[`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

## Usage

``` r
unflatten(tree, x)
```

## Arguments

- tree:

  (`RTree`)  
  Tree describing the target structure, as returned by
  [`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md).

- x:

  (`list`)  
  Flat list of leaf values, typically produced by
  [`flatten()`](https://r-xla.github.io/pjrt/dev/reference/flatten.md).

## Value

The reconstructed nested structure (list or single value).

## See also

[`flatten()`](https://r-xla.github.io/pjrt/dev/reference/flatten.md),
[`build_tree()`](https://r-xla.github.io/pjrt/dev/reference/build_tree.md)

## Examples

``` r
x <- list(a = 1, b = list(c = 2, d = 3))
tree <- build_tree(x)

unflatten(tree, flatten(x))
#> $a
#> [1] 1
#> 
#> $b
#> $b$c
#> [1] 2
#> 
#> $b$d
#> [1] 3
#> 
#> 
unflatten(tree, list(10, 20, 30))
#> $a
#> [1] 10
#> 
#> $b
#> $b$c
#> [1] 20
#> 
#> $b$d
#> [1] 30
#> 
#> 
```
