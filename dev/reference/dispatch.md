# Native eager-dispatch fast path

Dispatch a call through a
[`dispatcher()`](https://r-xla.github.io/pjrt/dev/reference/dispatcher.md)'s
executable cache, compiling on a miss. An input the dispatcher cannot
execute is an error, named after the offending argument; there is no
fallback for the caller to take.

## Usage

``` r
dispatch(dispatcher, args)

dispatcher_size(dispatcher)
```

## Arguments

- dispatcher:

  (`Dispatcher`)  
  A dispatcher from
  [`dispatcher()`](https://r-xla.github.io/pjrt/dev/reference/dispatcher.md).

- args:

  (`list`)  
  The (already evaluated) argument list of the call.

## Value

`dispatch()` returns the call's result. With `backend = "xla"` that is
the output buffers wrapped into `"AnvlArray"`s and re-nested by the
`compile` callback's `out_tree` (see
[`dispatcher()`](https://r-xla.github.io/pjrt/dev/reference/dispatcher.md));
with any other backend it is whatever the compiled closure returned.

`dispatcher_size()` returns the number of compiled executables the
dispatcher currently caches.
