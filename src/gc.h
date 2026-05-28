#pragma once

#include <cstddef>

namespace rpjrt {

// Run R's garbage collector, pending finalizers, and drain pjrt's deferred
// release queue. Called from the main R thread when a PJRT allocation fails
// with RESOURCE_EXHAUSTED so that unreferenced PJRTBuffer external pointers
// get destroyed (freeing their device memory) before we retry.
void call_r_gc();

// Number of times call_r_gc() has been invoked since process start. Exposed
// for tooling/tests to confirm the retry path actually fires.
std::size_t gc_call_count();

}  // namespace rpjrt
