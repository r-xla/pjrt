# Flatten

Recursively flattens a nested list into a single flat list containing
only the leaf values, preserving left-to-right order.

A bare (class-less) list is recursed into; `NULL` contributes no leaves;
an object of a class registered with
[`register_tree_node()`](https://r-xla.github.io/pjrt/reference/register_tree_node.md)
is recursed into through the children its `flatten` returns; everything
else (a classed object, an atomic, a function, ...) is a leaf.

Use
[`build_tree()`](https://r-xla.github.io/pjrt/reference/build_tree.md)
to capture the nesting structure so it can be restored with
[`unflatten()`](https://r-xla.github.io/pjrt/reference/unflatten.md).

## Usage

``` r
flatten(x, is_leaf = NULL)
```

## Arguments

- x:

  (any)  
  Object to flatten.

- is_leaf:

  (`function` \| `NULL`)  
  Called on every value the walk meets other than `NULL`, the list `x`
  included; a value it returns `TRUE` for is a leaf, even a list or an
  object of a registered class, as with JAX's
  `tree_flatten(is_leaf = )`. Must return `TRUE` or `FALSE`. `NULL`
  (default) recurses into every list and registered object.

## Value

[`list()`](https://rdrr.io/r/base/list.html) containing the flattened
values.

## See also

[`build_tree()`](https://r-xla.github.io/pjrt/reference/build_tree.md),
[`unflatten()`](https://r-xla.github.io/pjrt/reference/unflatten.md),
[`tree_size()`](https://r-xla.github.io/pjrt/reference/tree_size.md)

## Examples

``` r
x <- list(a = 1, b = list(c = 2, d = 3))
flatten(x)
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [1] 2
#> 
#> [[3]]
#> [1] 3
#> 

flatten(list(1:3, "hello"))
#> [[1]]
#> [1] 1 2 3
#> 
#> [[2]]
#> [1] "hello"
#> 

# stop at the list `b`
flatten(x, is_leaf = function(v) identical(v, x$b))
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [[2]]$c
#> [1] 2
#> 
#> [[2]]$d
#> [1] 3
#> 
#> 
```
