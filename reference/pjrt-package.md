# pjrt: R Interface to PJRT

Provides an R interface to PJRT (Pretty much Just another RunTime),
which allows you to run XLA or stableHLO programs on a variety of
hardware backends including CPU, GPU, and TPU.

## Environment Variables

- `TF_CPP_MIN_LOG_LEVEL`: Logging level for PJRT C++ API:

  - 0: shows info, warnings and errors

  - 1: shows warnings and errors

  - 2: shows errors

  - 3: shows nothing

- `PJRT_PLATFORM`: Default platform to use, falls back to `"cpu"`.

- `PJRT_CPU_DEVICE_COUNT`: The number of CPU devices to use. Defaults to
  1.

## See also

Useful links:

- <https://r-xla.github.io/pjrt/>

- <https://github.com/r-xla/pjrt>

- Report bugs at <https://github.com/r-xla/pjrt/issues>

## Author

**Maintainer**: Daniel Falbel <daniel@posit.co>
([ORCID](https://orcid.org/0009-0006-0143-2392))

Authors:

- Sebastian Fischer <seb.fischer@tutamail.com>
  ([ORCID](https://orcid.org/0000-0002-9609-3197))
