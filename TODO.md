# TODO

## Don't build CUDA kernels into installs that will never run them

`configure` decides whether to compile `src/cuda/*.cu` by asking "is there an
`nvcc` I can find?", never "will this installation use a GPU?". Those are not
the same question, and the gap goes one way: an install that has the `cuda12.8`
R package, `CUDA_HOME`, or a toolkit on the `PATH` for any reason at all gets
device code compiled into its `pjrt.so`, whether or not it will ever load the
CUDA plugin. Shared HPC login nodes, CI images and workstations that once
installed CUDA for something else all land in that bucket. (A genuinely bare
CPU machine gets nothing, so this is not every CPU install -- but it is more of
them than it should be.)

The cost today is small: one 57KB fatbin, plus a handful of `nvcc` invocations
at install time. It grows with each kernel added, and each kernel is compiled
for eight architectures.

Options, roughly in increasing order of how much they change:

- **An explicit switch.** `PJRT_CUDA_KERNELS=0/1` to force the decision, with
  the current search as the default. One line, fixes nothing on its own, but
  gives people building CPU-only images a way to opt out. Probably worth doing
  regardless of what else happens.
- **Gate on intent rather than on toolchain.** Reuse `cuda_available()` from
  `R/install.R` (Linux x86_64 plus a working `nvidia-smi`), which is already
  what `install_pjrt()` trusts to decide whether to fetch the CUDA plugin.
  Cheap and consistent, but wrong for the build-here-run-there case, which is
  exactly what container and cluster builds do.
- **Compile at run time with NVRTC instead.** `libnvrtc.so.12` ships in
  `cuda12.8` and compiles CUDA C++ to a cubin in-process, no `nvcc`. Verified
  working: our kernel compiles for the device actually present in ~40ms once
  libnvrtc is loaded, and the result is a 4.6KB cubin rather than a 57KB
  multi-architecture fatbin. This dissolves the problem instead of managing it
  -- CPU installs carry a kernel source string and nothing else, every install
  is byte-identical, and the SASS is targeted at the real device rather than
  JITted from PTX or picked out of a fatbin. The costs are real though:
  compile errors move from install time to first use (so CI has to cover what
  the compiler used to), the first `dlopen` of the 104MB libnvrtc is slow
  enough to notice, and the compile result wants an on-disk cache keyed on
  source hash + architecture + NVRTC version.
- **Put the kernels in a companion package.** This is how JAX sidesteps the
  problem entirely: its GPU kernels live in `jax-cuda12-plugin`, a separate
  wheel, so a CPU user never installs the device code in the first place and
  the decision is made by dependency resolution rather than by sniffing the
  build machine. The analogue here would be a `pjrt.cuda` package carrying
  `src/cuda/` and registering its handlers on load, which pjrt's deferred
  custom-call registry already supports without changes. The most invasive
  option, and it means a second package to release in lockstep, but it is the
  only one where the answer does not depend on what happens to be installed on
  the build machine.
- **Precompile once at release and ship the fatbin.** Removes `nvcc` from every
  install, but makes *everyone* carry the device code, so it trades this
  problem for a worse version of it. Only worth it if runtime compilation turns
  out not to be viable.

NVRTC looks like the right end state. It would also let `src/cuda/` kernels use
template parameters chosen at run time, which the ahead-of-time build cannot.
