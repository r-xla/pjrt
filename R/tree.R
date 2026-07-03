#' @title Flatten
#' @description
#' Recursively flattens a nested list into a single flat list containing only
#' the leaf values, preserving left-to-right order.
#'
#' A bare (class-less) list is recursed into; `NULL` contributes no leaves;
#' everything else (a classed object, an atomic, a function, ...) is a leaf.
#'
#' Use [build_tree()] to capture the nesting structure so it can be restored
#' with [unflatten()].
#' @param x (any)\cr
#'   Object to flatten.
#' @return `list()` containing the flattened values.
#' @seealso [build_tree()], [unflatten()], [tree_size()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' flatten(x)
#'
#' flatten(list(1:3, "hello"))
#' @export
flatten <- function(x) {
  impl_tree_flatten(x)
}

#' @title Build Tree
#' @description
#' Captures the nesting structure of an object as an opaque `PJRTNode` tree.
#' Each leaf in the input corresponds to its position in the flat list produced
#' by [flatten()]. The resulting tree can be passed to [unflatten()] to
#' reconstruct the original structure from a flat list.
#'
#' `NULL` is an empty node: it holds no leaf and contributes nothing to the
#' flat list, but is preserved in the tree (so `list(x, NULL)` and
#' `list(NULL, x)` have distinct trees).
#' @param x (any)\cr
#'   Object whose structure to capture. Bare lists are recursed into;
#'   everything else is a leaf.
#' @return A `PJRTNode`.
#' @seealso [flatten()], [unflatten()], [tree_size()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' tree <- build_tree(x)
#' tree_size(tree)
#'
#' unflatten(tree, flatten(x))
#' @export
build_tree <- function(x) {
  impl_tree_build(x)
}

#' @title Unflatten
#' @description
#' Reconstructs a nested structure from a flat list by using a tree previously
#' created with [build_tree()].
#' @param node (`PJRTNode`)\cr
#'   Tree describing the target structure, as returned by [build_tree()].
#' @param x (`list`)\cr
#'   Flat list of leaf values, typically produced by [flatten()].
#' @return The reconstructed nested structure (list or single value).
#' @seealso [flatten()], [build_tree()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' tree <- build_tree(x)
#'
#' unflatten(tree, flatten(x))
#' unflatten(tree, list(10, 20, 30))
#' @export
unflatten <- function(node, x) {
  impl_tree_unflatten(node, x)
}

#' @title Tree Size
#' @description
#' Counts the number of leaf nodes in a tree. This equals the length of the
#' flat list produced by [flatten()] on the original structure.
#' @param x (`PJRTNode`)\cr
#'   A tree as returned by [build_tree()].
#' @return A scalar `integer`.
#' @seealso [build_tree()], [flatten()]
#' @examples
#' tree_size(build_tree(list(a = 1, b = list(c = 2, d = 3))))
#' @export
tree_size <- function(x) {
  impl_tree_size(x)
}

#' @title Compare Tree Structures
#' @description
#' Structural equality of two trees: identical node kinds, child structure,
#' leaf positions, and names.
#' @param a,b (`PJRTNode`)\cr
#'   Trees to compare, as returned by [build_tree()].
#' @return A scalar `logical`.
#' @seealso [tree_diff()] to locate the first divergence.
#' @examples
#' tree_equal(build_tree(list(a = 1)), build_tree(list(a = 2)))
#' tree_equal(build_tree(list(a = 1)), build_tree(list(b = 1)))
#' @export
tree_equal <- function(a, b) {
  impl_tree_equal(a, b)
}

#' @title Tree Node Kind
#' @description
#' The kind of a tree's root node: `"leaf"` for a single leaf, `"list"` for a
#' list node, or `"null"` for the empty (`NULL`) node.
#' @param x (`PJRTNode`)\cr
#'   A tree as returned by [build_tree()].
#' @return A scalar `character`.
#' @examples
#' tree_kind(build_tree(1))
#' tree_kind(build_tree(list(1)))
#' tree_kind(build_tree(NULL))
#' @export
tree_kind <- function(x) {
  impl_tree_kind(x)
}

#' @title Top-Level Child Names
#' @description
#' The names of a list tree's top-level children (with `""` for unnamed
#' slots), or `NULL` if the root is not a list node or carries no names.
#' @param x (`PJRTNode`)\cr
#'   A tree as returned by [build_tree()].
#' @return A `character` vector or `NULL`.
#' @examples
#' tree_names(build_tree(list(a = 1, b = 2)))
#' tree_names(build_tree(list(1, 2)))
#' @export
tree_names <- function(x) {
  impl_tree_names(x)
}

#' @title Top-Level Child Kinds
#' @description
#' The node kind (`"leaf"`, `"list"`, or `"null"`) of each top-level child of
#' a list tree.
#' @param x (`PJRTNode`)\cr
#'   A tree whose root is a list node.
#' @return A `character` vector, one element per top-level child.
#' @examples
#' child_kinds(build_tree(list(1, list(2), NULL)))
#' @export
child_kinds <- function(x) {
  impl_tree_child_kinds(x)
}

#' @title Top-Level Child Sizes
#' @description
#' The number of leaves under each top-level child of a list tree.
#' @param x (`PJRTNode`)\cr
#'   A tree whose root is a list node.
#' @return An `integer` vector, one element per top-level child.
#' @examples
#' child_sizes(build_tree(list(a = 1, b = list(2, 3), c = NULL)))
#' @export
child_sizes <- function(x) {
  impl_tree_child_sizes(x)
}

#' @title Per-Leaf Top-Level Names
#' @description
#' For each leaf (in flat order), the name of the top-level child it sits
#' under (`""` for unnamed children).
#' @param x (`PJRTNode`)\cr
#'   A tree whose root is a list node.
#' @return A `character` vector of length `tree_size(x)`.
#' @examples
#' flat_names(build_tree(list(a = 1, b = list(2, 3))))
#' @export
flat_names <- function(x) {
  impl_tree_flat_names(x)
}

#' @title Tree Path
#' @description
#' Returns the human-readable path for a single leaf identified by its flat
#' index (e.g. `"a$b"` or `"[[2]]"`). Only descends into the branch containing
#' the target leaf, making it efficient for error reporting.
#' @param node (`PJRTNode`)\cr
#'   A tree as returned by [build_tree()].
#' @param i (`integer(1)`)\cr
#'   The flat index of the leaf.
#' @return A scalar `character` string (`""` for a leaf at the root).
#' @seealso [build_tree()], [flatten()]
#' @examples
#' tree_path(build_tree(list(a = 1, b = list(c = 2))), 2L)
#' @export
tree_path <- function(node, i) {
  impl_tree_path(node, as.integer(i))
}

#' @title Filter a Tree by Top-Level Names
#' @description
#' Keeps only the top-level children whose names match `names` (in tree
#' order), reindexing the remaining leaves so they map to contiguous positions
#' in a flat list.
#' @param tree (`PJRTNode`)\cr
#'   A named list tree as returned by [build_tree()].
#' @param names (`character()`)\cr
#'   Names of children to keep.
#' @return A `PJRTNode` containing only the selected children.
#' @seealso [build_tree()], [unflatten()]
#' @examples
#' x <- list(a = 1, b = 2, c = 3)
#' sub <- filter_by_names(build_tree(x), c("a", "c"))
#' tree_size(sub)
#' unflatten(sub, x[c("a", "c")])
#' @export
filter_by_names <- function(tree, names) {
  impl_tree_filter_by_names(tree, as.character(names))
}

#' @title Concatenate Trees Under a New Root
#' @description
#' Builds a new list tree whose children are the given trees, reassigning leaf
#' indices contiguously across all children (in order).
#' @param nodes (`list` of `PJRTNode`)\cr
#'   The child trees.
#' @param names (`NULL` | `character()`)\cr
#'   Optional names for the children.
#' @return A `PJRTNode`.
#' @examples
#' combined <- tree_concat(
#'   list(build_tree(list(1, 2)), build_tree(3)),
#'   names = c("value", "grad")
#' )
#' tree_size(combined)
#' @export
tree_concat <- function(nodes, names = NULL) {
  checkmate::assert_list(nodes)
  impl_tree_concat(nodes, names)
}

#' @title Flat Mask From Top-Level Names
#' @description
#' A logical mask over the leaves (in flat order) that is `TRUE` for every
#' leaf sitting under a top-level child whose name is in `names`. An empty
#' `names` yields an all-`FALSE` mask.
#' @param tree (`PJRTNode`)\cr
#'   A tree whose root is a list node.
#' @param names (`character()`)\cr
#'   Top-level child names to mark.
#' @return A `logical` vector of length `tree_size(tree)`.
#' @examples
#' mask_from_names(build_tree(list(a = 1, b = list(2, 3))), "b")
#' @export
mask_from_names <- function(tree, names) {
  impl_tree_mask_from_names(tree, as.character(names))
}

#' @title Canonical Tree Representation
#' @description
#' A canonical structural string for a tree: `"*"` for a leaf, `"NULL"` for an
#' empty node, and `"list(...)"` for list nodes. Two trees are structurally
#' equal iff their representations are equal, so the string can serve as a
#' value-semantics cache key for a tree.
#' @param x (`PJRTNode`)\cr
#'   A tree as returned by [build_tree()].
#' @return A scalar `character` string.
#' @examples
#' tree_repr(build_tree(list(a = 1, b = list(2, NULL))))
#' @export
tree_repr <- function(x) {
  impl_tree_repr(x)
}

#' @title Difference Between Trees
#' @description
#' Walks two trees in parallel and returns the first path/subtree pair where
#' they diverge, or `NULL` if they are structurally identical. The returned
#' `prefix` follows [tree_path()] syntax; `a` and `b` are the [tree_repr()]
#' strings of the diverging subtrees.
#' @param a,b (`PJRTNode`)\cr
#'   Trees to compare, as returned by [build_tree()].
#' @return `NULL` if `a` and `b` are structurally identical, otherwise a list
#'   with elements `prefix`, `a`, and `b`.
#' @seealso [build_tree()], [tree_path()], [tree_equal()]
#' @examples
#' tree_diff(build_tree(list(a = 1)), build_tree(list(a = list(1, 2))))
#' @export
tree_diff <- function(a, b) {
  impl_tree_diff(a, b)
}

#' @title Map Over a Tree
#' @description
#' Apply a function to each leaf of a (possibly nested) list, preserving the
#' tree structure. Equivalent to flattening `.x` with [flatten()], applying
#' `.f` to each leaf, and reassembling with [unflatten()].
#' @param .x (any)\cr
#'   A leaf or a (nested) list of leaves.
#' @param .f (`function`)\cr
#'   Function to apply to each leaf of `.x`.
#' @param ... Additional arguments passed to `.f`.
#' @return An object with the same nesting structure as `.x`, where each leaf
#'   is the result of `.f(leaf, ...)`.
#' @seealso [flatten()], [build_tree()], [unflatten()]
#' @examples
#' map_tree(list(a = 1, b = list(c = 2, d = 3)), \(x) x + 1)
#' @export
map_tree <- function(.x, .f, ...) {
  tree <- build_tree(.x)
  flat <- flatten(.x)
  result <- lapply(seq_along(flat), function(i) {
    tryCatch(
      .f(flat[[i]], ...),
      error = function(e) {
        path <- tree_path(tree, i)
        loc <- if (nzchar(path)) path else "<root>"
        cli::cli_abort(
          "Error applying {.arg .f} to leaf at {.code {loc}}.",
          parent = e,
          call = NULL
        )
      }
    )
  })
  unflatten(tree, result)
}

#' @title Map Over Multiple Trees
#' @description
#' Apply a function leaf-wise over several trees with the same structure.
#' All trees in `.l` must have identical structure.
#' @param .l (`list`)\cr
#'   A non-empty list of trees, all with the same structure.
#' @param .f (`function`)\cr
#'   Function to call with one leaf from each tree (positional arguments, in
#'   the order given by `.l`).
#' @param ... Additional arguments passed to `.f` after the per-tree leaves.
#' @return A tree with the same structure as `.l[[1]]`, where each leaf is
#'   `.f(leaf_1, leaf_2, ..., leaf_n, ...)`.
#' @seealso [map_tree()], [flatten()], [unflatten()]
#' @examples
#' pmap_tree(list(list(a = 1, b = 2), list(a = 10, b = 20)), `+`)
#' @export
pmap_tree <- function(.l, .f, ...) {
  if (!is.list(.l) || length(.l) == 0L) {
    cli::cli_abort("{.arg .l} must be a non-empty list of trees.")
  }
  tree <- build_tree(.l[[1L]])
  for (i in seq_along(.l)[-1L]) {
    other <- build_tree(.l[[i]])
    if (!tree_equal(tree, other)) {
      diff <- tree_diff(tree, other)
      header <- if (nzchar(diff$prefix)) {
        "First mismatch at {.code {diff$prefix}}:"
      } else {
        "Trees differ at the root:"
      }
      cli::cli_abort(c(
        "All trees in {.arg .l} must have the same structure.",
        "x" = header,
        "*" = "{.code .l[[1]]}: {diff$a}",
        "*" = "{.code .l[[{i}]]}: {diff$b}"
      ))
    }
  }
  flats <- lapply(.l, flatten)
  n <- length(flats[[1L]])
  result <- lapply(seq_len(n), function(i) {
    tryCatch(
      do.call(.f, c(lapply(flats, `[[`, i), list(...))),
      error = function(e) {
        path <- tree_path(tree, i)
        loc <- if (nzchar(path)) path else "<root>"
        cli::cli_abort(
          "Error applying {.arg .f} to leaves at {.code {loc}}.",
          parent = e,
          call = NULL
        )
      }
    )
  })
  unflatten(tree, result)
}

#' @title Flatten a Function
#' @description
#' Wraps a function so it can be called with the flat leaf list of its
#' original (nested) arguments. The wrapper unflattens the leaves back into
#' the original argument structure, calls `f`, and returns
#' `list(build_tree(out), flatten(out))`.
#' @param f (`function`)\cr
#'   Function to wrap.
#' @param ... Example arguments whose structure the flat inputs follow.
#'   Mutually exclusive with `in_node`.
#' @param in_node (`NULL` | `PJRTNode`)\cr
#'   The input structure, if already available.
#' @return A `FlattenedFunction`.
#' @seealso [build_tree()], [flatten()], [unflatten()]
#' @examples
#' f <- function(x) list(sum = x$a + x$b)
#' ff <- flatten_fun(f, list(a = 1, b = 2))
#' ff(3, 4)
#' @export
flatten_fun <- function(f, ..., in_node = NULL) {
  if (is.null(in_node)) {
    in_node <- build_tree(list(...))
  } else if (...length()) {
    cli::cli_abort("in_node is not compatible with ... arguments")
  }
  f_orig <- f
  f <- function(...) {
    args <- unflatten(in_node, list(...))
    outs <- do.call(f_orig, args)
    list(
      build_tree(outs),
      flatten(outs)
    )
  }
  class(f) <- "FlattenedFunction"
  f
}

#' @export
format.PJRTNode <- function(x, ...) {
  tree_repr(x)
}

#' @export
print.PJRTNode <- function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}
