# pjrt: R Interface to PJRT

Provides an R interface to PJRT (Pretty much Just another RunTime),
which allows you to run XLA or stableHLO programs on a variety of
hardware backends including CPU, GPU, and TPU.

## Environment Variables

**Configuration options provided by XLA**

XLA provides various configuration options, but their documentation is
scattered across various websites. The options include:

- `TF_CPP_MIN_LOG_LEVEL`: Logging level for PJRT C++ API:

  - 0: shows info, warnings and errors

  - 1: shows warnings and errors

  - 2: shows errors

  - 3: shows nothing

- `XLA_FLAGS`: See the [openxla
  website](https://openxla.org/xla/flags_guidance) for more information.

**Configuration options provided by this package**

- `PJRT_PLATFORM`: Default platform to use, falls back to `"cpu"`.

- `PJRT_PLUGIN_PATH_<PLATFORM>`: Path to custom plugin library file for
  a specific platform (e.g., `PJRT_PLUGIN_PATH_CPU`,
  `PJRT_PLUGIN_PATH_CUDA`, `PJRT_PLUGIN_PATH_METAL`). If set, the
  package will use this path instead of downloading the plugin.

- `PJRT_PLUGIN_URL_<PLATFORM>`: URL to download plugin from for a
  specific platform (e.g., `PJRT_PLUGIN_URL_CPU`,
  `PJRT_PLUGIN_URL_CUDA`, `PJRT_PLUGIN_URL_METAL`). If set, overrides
  the default plugin download URL.

- `PJRT_ZML_ARTIFACT_VERSION`: Version of ZML artifacts to download.
  Only used when downloading plugins from zml/pjrt-artifacts.

- `PJRT_CPU_DEVICE_COUNT`: The number of CPU devices to use. Defaults
  to 1. This is primarily intended for testing purposes.

## See also

Useful links:

- <https://r-xla.github.io/pjrt/>

- <https://github.com/r-xla/pjrt>

- Report bugs at <https://github.com/r-xla/pjrt/issues>

## Author

**Maintainer**: Sebastian Fischer <seb.fischer@tutamail.com>
([ORCID](https://orcid.org/0000-0002-9609-3197))

Authors:

- Daniel Falbel <daniel@posit.co>
  ([ORCID](https://orcid.org/0009-0006-0143-2392))
