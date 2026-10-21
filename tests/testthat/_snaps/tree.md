# print.RTree / prints the structure of a tree

    Code
      build_tree(1)
    Output
      *
    Code
      build_tree(NULL)
    Output
      NULL
    Code
      build_tree(list())
    Output
      list()
    Code
      build_tree(structure(list(), names = character(0L)))
    Output
      list<named>()
    Code
      build_tree(list(1, NULL, list(2, 3)))
    Output
      list(*, NULL, list(*, *))
    Code
      build_tree(list(a = 1, b = list(c = 2, d = NULL)))
    Output
      list<named>(a = *, b = list<named>(c = *, d = NULL))
    Code
      build_tree(list(a = 1, 2))
    Output
      list<named>(a = *, *)
    Code
      tree_concat(list(build_tree(list(1, 2)), build_tree(3)), names = c("value",
        "grad"))
    Output
      list<named>(value = list(*, *), grad = *)

# register_tree_node / prints an object node with its class and label

    Code
      build_tree(point(1, 2))
    Output
      <point: cm><named>(x = *, y = *)
    Code
      build_tree(list(a = point(1, list(2, NULL)), b = 3))
    Output
      list<named>(a = <point: cm><named>(x = *, y = list(*, NULL)), b = *)
    Code
      build_tree(point(point(1, 2, "in"), point(3, 4, "mm"), "km"))
    Output
      <point: km><named>(x = <point: in><named>(x = *, y = *), y = <point: mm><named>(x = *, y = *))
    Code
      build_tree(point(1, 2, NULL))
    Output
      <point><named>(x = *, y = *)
    Code
      tree_diff(build_tree(list(p = point(1, 2, "cm"))), build_tree(list(p = point(1,
        2, "in"))))
    Output
      $prefix
      [1] "p"
      
      $a
      [1] "<point: cm><named>(x = *, y = *)"
      
      $b
      [1] "<point: in><named>(x = *, y = *)"
      

# register_tree_node / prints an object node without a label once its class is unregistered

    Code
      tree
    Output
      <point><named>(x = *, y = *)

# register_tree_node / requires flatten to return list(children, aux)

    Code
      flatten(point(1, 2))
    Condition
      Error:
      ! the `flatten` registered for class "point" must return `list(children, aux)`, where `children` is a list

---

    Code
      flatten(point(1, 2))
    Condition
      Error:
      ! the `flatten` registered for class "point" must return `list(children, aux)`, where `children` is a list

# register_tree_node / requires repr to return a string

    Code
      format(build_tree(point(1, 2)))
    Condition
      Error:
      ! the `repr` registered for class "point" must return a string

# register_tree_node / validates its arguments

    Code
      register_tree_node("point", identity, identity)
    Condition
      Error in `checkmate::assert_function()`:
      ! argument "repr" is missing, with no default

---

    Code
      register_tree_node(c("a", "b"), identity, identity, identity)
    Condition
      Error in `register_tree_node()`:
      ! Assertion on 'class' failed: Must have length 1.

