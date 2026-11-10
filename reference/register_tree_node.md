# Register a Class as a Tree Node

Makes the objects of a class nodes of a tree rather than leaves:
[`flatten()`](https://r-xla.github.io/pjrt/reference/flatten.md) and
[`build_tree()`](https://r-xla.github.io/pjrt/reference/build_tree.md)
recurse into the children `flatten` returns,
[`unflatten()`](https://r-xla.github.io/pjrt/reference/unflatten.md)
rebuilds an object of the class from them with `unflatten`, and
[`tree_repr()`](https://r-xla.github.io/pjrt/reference/tree_repr.md)
(and so the tree's printer) renders the node with `repr`. This is JAX's
`register_pytree_node()`, plus the printer.

Besides its children, `flatten` returns the object's aux data: whatever
else `unflatten` needs to rebuild it, such as a setting that is not an
array. The class and the aux data are part of the tree's structure,
compared with [`identical()`](https://rdrr.io/r/base/identical.html), so
two trees that differ in either are not equal and the dispatcher
compiles them separately. Keep the aux data small and put the arrays
among the children.

An object is a node of the first of its classes that is registered. A
registered class cannot be registered again: to change its functions,
unregister it with `unregister_tree_node()` and register it anew. A tree
keeps the registration it was built under: it unflattens and prints its
objects with that registration's functions even after the class is
unregistered, and it is not equal to a tree built under another
registration of the class, so the dispatcher never serves an entry
compiled under the old functions.

## Usage

``` r
register_tree_node(class, flatten, unflatten, repr)

unregister_tree_node(class)
```

## Arguments

- class:

  (`character(1)`)  
  The class.

- flatten:

  (`function`)  
  Takes an object of the class and returns `list(children, aux)`: its
  children, a bare `list` whose names the tree keeps, and its aux data,
  any value (`NULL` if there is none).

- unflatten:

  (`function`)  
  Takes the children, a `list` named as `flatten` names them, and the
  aux data, and returns an object of the class.

- repr:

  (`function`)  
  Takes the children's
  [`tree_repr()`](https://r-xla.github.io/pjrt/reference/tree_repr.md)
  strings, a `character` vector named as `flatten` names the children,
  and the aux data, and returns the node's string, a `character(1)`,
  which
  [`tree_repr()`](https://r-xla.github.io/pjrt/reference/tree_repr.md)
  shows as it is. Render nodes that differ in their aux data or their
  children's names differently, so that trees that are not equal print
  differently.

## Value

`NULL`, invisibly.

## See also

[`flatten()`](https://r-xla.github.io/pjrt/reference/flatten.md),
[`build_tree()`](https://r-xla.github.io/pjrt/reference/build_tree.md)

## Examples

``` r
point <- function(x, y, unit) {
  structure(list(x = x, y = y, unit = unit), class = "point")
}
register_tree_node(
  "point",
  flatten = function(p) list(list(x = p$x, y = p$y), p$unit),
  unflatten = function(children, unit) point(children$x, children$y, unit),
  repr = function(children, unit) {
    sprintf("point[%s](%s)", unit, paste(children, collapse = ", "))
  }
)
flatten(list(point(1, 2, "cm"), 3))
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [1] 2
#> 
#> [[3]]
#> [1] 3
#> 
tree <- build_tree(list(point(1, 2, "cm"), 3))
tree
#> list(point[cm](*, *), *)
unflatten(tree, list(10, 20, 30))
#> [[1]]
#> $x
#> [1] 10
#> 
#> $y
#> [1] 20
#> 
#> $unit
#> [1] "cm"
#> 
#> attr(,"class")
#> [1] "point"
#> 
#> [[2]]
#> [1] 30
#> 
build_tree(point(1, 2, "cm")) == build_tree(point(1, 2, "in"))
#> [1] FALSE
unregister_tree_node("point")
```
