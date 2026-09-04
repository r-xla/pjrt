// The dispatcher's cache key, and the dtype vocabulary it is built from.
//
// Split out of dispatch.cpp so that test-dispatch.cpp can exercise it directly:
// these are pure data plus a hash and an equality, and they are where a mistake
// silently returns the wrong compiled program.

#pragma once

#include <Rcpp.h>
#include <Rversion.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "buffer.h"
#include "hash.h"
#include "tree.h"

namespace rpjrt {

// The dtype of an anvl array, whatever backend holds it. Deliberately a type of
// our own rather than PJRT_Buffer_Type: an Aval describes a plain R array that
// never went near PJRT just as readily as a PJRT buffer.
//
// The set is what the dispatcher can represent, which is also exactly what
// pjrt's string_to_pjrt_buffer_type() accepts. tengen now names more dtypes
// than this (f16, bf16, f8*, complex, sub-byte ints); those are rejected by
// anvl_dtype_from_tengen() below rather than keyed approximately. Conversions
// in either direction are explicit switches rather than casts, so a PJRT type
// outside this set maps to kInvalid rather than silently becoming a
// neighbouring dtype.
enum class AnvlDtype {
  kInvalid,
  kBool,
  kI8,
  kI16,
  kI32,
  kI64,
  kU8,
  kU16,
  kU32,
  kU64,
  kF32,
  kF64,
  // The R storage types, which a bare R leaf (AvalKind::kRData) is keyed by.
  // These are not device dtypes and no buffer ever holds one: a bare R value
  // has no dtype until the program says what it is uploaded as, so its key
  // entry names what the value *is* rather than a dtype it is not yet. They
  // are spelled as R's own typeof() values, which is why "double" is not f64
  // and "integer" not i32 -- neither says anything about a buffer's width.
  kDouble,
  kInteger,
  kLogical,
};

inline AnvlDtype anvl_dtype_from_pjrt(PJRT_Buffer_Type t) {
  switch (t) {
    case PJRT_Buffer_Type_PRED:
      return AnvlDtype::kBool;
    case PJRT_Buffer_Type_S8:
      return AnvlDtype::kI8;
    case PJRT_Buffer_Type_S16:
      return AnvlDtype::kI16;
    case PJRT_Buffer_Type_S32:
      return AnvlDtype::kI32;
    case PJRT_Buffer_Type_S64:
      return AnvlDtype::kI64;
    case PJRT_Buffer_Type_U8:
      return AnvlDtype::kU8;
    case PJRT_Buffer_Type_U16:
      return AnvlDtype::kU16;
    case PJRT_Buffer_Type_U32:
      return AnvlDtype::kU32;
    case PJRT_Buffer_Type_U64:
      return AnvlDtype::kU64;
    case PJRT_Buffer_Type_F32:
      return AnvlDtype::kF32;
    case PJRT_Buffer_Type_F64:
      return AnvlDtype::kF64;
    default:
      return AnvlDtype::kInvalid;
  }
}

// The canonical name, as tengen spells it -- this is the vocabulary that
// crosses into R (the compile callback's avals). The boolean type is the one
// place the two layers disagree: tengen calls it "bool" and pjrt's own C-API
// layer calls it "pred", so the buffer-facing code (string_to_pjrt_buffer_type
// and friends) keeps saying "pred" and translates at its edge.
inline const char *anvl_dtype_name(AnvlDtype d) {
  switch (d) {
    case AnvlDtype::kBool:
      return "bool";
    case AnvlDtype::kI8:
      return "i8";
    case AnvlDtype::kI16:
      return "i16";
    case AnvlDtype::kI32:
      return "i32";
    case AnvlDtype::kI64:
      return "i64";
    case AnvlDtype::kU8:
      return "ui8";
    case AnvlDtype::kU16:
      return "ui16";
    case AnvlDtype::kU32:
      return "ui32";
    case AnvlDtype::kU64:
      return "ui64";
    case AnvlDtype::kF32:
      return "f32";
    case AnvlDtype::kF64:
      return "f64";
    case AnvlDtype::kDouble:
      return "double";
    case AnvlDtype::kInteger:
      return "integer";
    case AnvlDtype::kLogical:
      return "logical";
    case AnvlDtype::kInvalid:
      return "invalid";
  }
  return "invalid";
}

// Translate a canonical dtype name to an AnvlDtype. tengen names more dtypes
// than the dispatcher supports (f16, bf16, f8*, complex, sub-byte ints); those
// yield kInvalid and the caller rejects them rather than keying approximately.
// The R storage types ("double", ...) are deliberately absent: this parses the
// strings that name a *buffer's* type (a tengen DataType, the compile
// callback's `input_dtypes`), and no buffer is ever of an R storage type.
inline AnvlDtype anvl_dtype_from_name(const char *name) {
  if (!std::strcmp(name, "bool")) return AnvlDtype::kBool;
  if (!std::strcmp(name, "i8")) return AnvlDtype::kI8;
  if (!std::strcmp(name, "i16")) return AnvlDtype::kI16;
  if (!std::strcmp(name, "i32")) return AnvlDtype::kI32;
  if (!std::strcmp(name, "i64")) return AnvlDtype::kI64;
  if (!std::strcmp(name, "ui8")) return AnvlDtype::kU8;
  if (!std::strcmp(name, "ui16")) return AnvlDtype::kU16;
  if (!std::strcmp(name, "ui32")) return AnvlDtype::kU32;
  if (!std::strcmp(name, "ui64")) return AnvlDtype::kU64;
  if (!std::strcmp(name, "f32")) return AnvlDtype::kF32;
  if (!std::strcmp(name, "f64")) return AnvlDtype::kF64;
  return AnvlDtype::kInvalid;
}

// The same, for a tengen DataType object: a length-1 character vector classed
// "DataType" whose string is the canonical dtype name.
inline AnvlDtype anvl_dtype_from_tengen(SEXP dtype) {
  if (TYPEOF(dtype) != STRSXP || XLENGTH(dtype) != 1) {
    return AnvlDtype::kInvalid;
  }
  if (!Rf_inherits(dtype, "DataType")) {
    return AnvlDtype::kInvalid;
  }
  return anvl_dtype_from_name(CHAR(STRING_ELT(dtype, 0)));
}

// What kind of value an Aval abstracts. The two differ in where execution finds
// the value *and* in what the caller compiles for it, so an Aval is a variant
// over them rather than a plain (dtype, shape) pair:
//   kArray  one of the backend's arrays. Its dtype is its own; execution hands
//           the program the array's `$data`.
//   kRData  a bare R literal or array. It has no dtype of its own until the
//           program says what it is used as (anvl's RData values, which let
//           `x_f64 / sqrt(2)` see the exact double rather than one rounded
//           through f32), so `dtype` is its R storage type -- "double",
//           "integer" or "logical" -- rather than any device dtype: what the
//           leaf is uploaded at is the entry's `input_dtypes`, which the
//           callback must declare. Execution uploads the leaf itself.
enum class AvalKind : std::uint8_t { kArray, kRData };

// Per-leaf abstract value -- mirrors anvl's nv_aval(dtype, shape) and its
// RDataArray. All three fields are read off the leaf. The device is not part of
// it: it is a single per-call value on the CacheKey.
struct Aval {
  AvalKind kind = AvalKind::kArray;
  AnvlDtype dtype = AnvlDtype::kInvalid;
  std::vector<int64_t> shape;
};

inline const char *aval_kind_name(AvalKind k) {
  return k == AvalKind::kRData ? "rdata" : "array";
}

inline std::uint64_t aval_hash(const Aval &a) {
  std::uint64_t h = static_cast<std::uint64_t>(a.kind);
  h = hash_combine(h, static_cast<std::uint64_t>(a.dtype));
  for (int64_t d : a.shape) {
    h = hash_combine(h, static_cast<std::uint64_t>(d));
  }
  return h;
}

inline bool aval_eq(const Aval &a, const Aval &b) {
  return a.kind == b.kind && a.dtype == b.dtype && a.shape == b.shape;
}

// identical(), tightened for use as a cache key.
//
// IDENT_USE_CLOENV compares closure environments (R's default); without it two
// distinct closures with the same body would wrongly merge.
//
// IDENT_NUM_AS_BITS compares doubles and complex bitwise rather than with `==`.
// R's default merges +0.0 with -0.0, and bit64 stores NA_integer64_ as the
// int64 minimum -- whose double reinterpretation is -0.0 -- so under `==` a
// static NA_integer64_ and a static 0 are "identical" and would share a cache
// entry, silently running each other's executable. Comparing bits splits them.
// The cost is only that +0.0 and -0.0 now compile separate (identical) entries:
// a finer key can waste a compile, never return the wrong program.
inline bool r_identical(SEXP a, SEXP b) {
  return R_compute_identical(a, b, IDENT_USE_CLOENV | IDENT_NUM_AS_BITS);
}

// A device token: the address of a *canonical* device object. Never
// dereferenced, only compared and folded, so one `const void*` identifies a
// device of any backend -- the key needs no identical(), no variant type, and
// no per-backend branch in its hash or equality.
//
// How a device object maps to its canonical representative is the engine's
// business (Engine::canonical_device()). The canonical objects are preserved
// for the dispatcher's lifetime, which is what keeps a token's address stable
// and unambiguous.
using DeviceToken = const void *;

// One leaf of the cache key. A leaf is either static -- keyed by value via
// r_identical(), and excluded from execution, because a static is baked into
// the executable as a constant -- or dynamic, in which case it is keyed by its
// Aval and supplied at execute time. A leaf that is neither is not a valid
// input, and impl_dispatch_run()'s classification loop rejects it -- naming the
// offending argument -- before any key is built.
//
// What sort of dynamic leaf it is lives in the Aval (see AvalKind), which is
// also what keeps `f(x, y)` and `f(x, 1)` two entries: their avals differ in
// kind, so they compile to two programs.
struct KeyLeaf {
  bool is_static = false;
  Aval aval;                // dynamic leaf
  SEXP value = R_NilValue;  // static leaf: the leaf itself
};

// A closure's formals pairlist. The only R-version-dependent call in the
// package: R_ClosureFormals() is API from R 4.5.0, and the FORMALS() it
// replaced was dropped from Rinternals.h in 4.6, so neither spelling spans both
// and the choice has to be made at compile time.
inline SEXP closure_formals(SEXP f) {
#if defined(R_VERSION) && R_VERSION >= R_Version(4, 5, 0)
  return R_ClosureFormals(f);
#else
  return FORMALS(f);
#endif
}

// Fold a closure static: its formal names, then its body as R would print it.
// Lives here rather than in hash_atomic() because it is a cache-key concern --
// hash_atomic() folds vector contents, and a closure has none.
//
// Sound by the same rule as every other fold: identical() compares a closure's
// formals, body and environment, so two closures it calls equal necessarily
// agree on the first two and fold alike. The fold can therefore only collide,
// never split.
//
// R_ClosureExpr() rather than BODY(): R byte-compiles closures, and a compiled
// one's BODY is a BCODESXP; only this decodes it back to the source expression,
// so `f` and a byte-compiled copy of it fold alike. Coercing that expression to
// STRSXP is the C-level as.character(): a call yields its deparsed elements, a
// symbol or a literal body yields itself. Anything else folds nothing.
inline std::uint64_t hash_closure(std::uint64_t h, SEXP f) {
  // names() of the formals pairlist: its tags, as a STRSXP. Shield rather than
  // PROTECT/UNPROTECT: hash_atomic() can throw, and an RAII guard unwinds the
  // protect stack where a bare UNPROTECT would be skipped.
  Rcpp::Shield<SEXP> nms(Rf_getAttrib(closure_formals(f), R_NamesSymbol));
  h = hash_atomic(h, nms);
  SEXP body = R_ClosureExpr(f);
  if (TYPEOF(body) == LANGSXP || TYPEOF(body) == SYMSXP) {
    Rcpp::Shield<SEXP> chars(Rf_coerceVector(body, STRSXP));
    return hash_atomic(h, chars);
  }
  // A literal body (`function() 1`) is already atomic; anything else folds
  // nothing and identical() decides.
  return hash_atomic(h, body);
}

// The executable-cache key -- mirrors anvl's list(in_tree, key_leaves, device).
// There is no backend component: a dispatcher accepts arrays of exactly one
// backend and owns its own cache, so no two keys of one cache could ever differ
// in it.
struct CacheKey {
  RTree in_tree;
  std::vector<KeyLeaf> leaves;
  DeviceToken device = nullptr;
  // Key material the caller resolves per call (?dispatcher's `context`): what
  // the compiled program depends on beyond its inputs -- anvl's default dtypes,
  // say. Empty when the dispatcher was created without a resolver.
  std::vector<std::string> context;
};

// CacheKeyHash and CacheKeyEq are functors rather than plain functions because
// unordered_map -- and LRUCache, which forwards them -- take the Hash and Eq as
// template *type* parameters. Passing them as types lets the map
// default-construct them and inline each call.
struct CacheKeyHash {
  // unordered_map's Hash concept requires std::size_t, so the 64-bit
  // accumulator is narrowed on return (a no-op on the 64-bit platforms we build
  // for).
  std::size_t operator()(const CacheKey &k) const {
    std::uint64_t h = tree_hash(k.in_tree);
    h = hash_combine(h, reinterpret_cast<std::uintptr_t>(k.device));
    // Length first, so ("ab") and ("a", "b") cannot fold to one stream.
    h = hash_combine(h, k.context.size());
    for (const std::string &s : k.context) {
      h = hash_combine(h, std::hash<std::string>{}(s));
    }
    h = hash_combine(h, k.leaves.size());
    for (const KeyLeaf &leaf : k.leaves) {
      // Folded before the per-leaf material, so a value-keyed leaf's hash
      // stream can never coincide with an Aval-keyed one's: the domain
      // separator. The Aval's own kind is folded by aval_hash(), so an array
      // and an rdata leaf of the same dtype and shape land in different
      // buckets, as CacheKeyEq requires.
      h = hash_combine(h, leaf.is_static ? 1u : 0u);
      if (!leaf.is_static) {
        h = hash_combine(h, aval_hash(leaf.aval));
        continue;
      }
      // Exact equality is r_identical(); folding a leaf's contents keeps that
      // call off the common path, where two static values (a TRUE and a FALSE,
      // say) would otherwise share type, length, and therefore bucket.
      //
      // Atomics fold their contents (hash_atomic), closures their formals and
      // body (hash_closure). A static of any other type -- a list, an
      // environment -- folds nothing and is separated only by its type and
      // length here, and by r_identical() in CacheKeyEq. That is conservative,
      // never wrong: a coarser hash costs a bucket collision, never a wrong
      // cache hit.
      h = hash_combine(h, static_cast<std::uint64_t>(TYPEOF(leaf.value)));
      h = hash_combine(h, static_cast<std::uint64_t>(Rf_xlength(leaf.value)));
      if (TYPEOF(leaf.value) == CLOSXP) {
        h = hash_closure(h, leaf.value);
      } else {
        h = hash_atomic(h, leaf.value);
      }
    }
    return static_cast<std::size_t>(h);
  }
};

struct CacheKeyEq {
  bool operator()(const CacheKey &a, const CacheKey &b) const {
    if (!tree_eq(a.in_tree, b.in_tree)) return false;
    if (a.device != b.device) return false;
    if (a.context != b.context) return false;
    if (a.leaves.size() != b.leaves.size()) return false;
    for (std::size_t k = 0; k < a.leaves.size(); ++k) {
      const KeyLeaf &x = a.leaves[k];
      const KeyLeaf &y = b.leaves[k];
      // Static-ness is already ruled out by tree_eq, which compares the
      // argument names it follows; the array/rdata split is not, and aval_eq()
      // is what rules it out -- two leaves of the same dtype and shape but
      // different kind compile to different programs.
      if (x.is_static != y.is_static) return false;
      if (x.is_static) {
        if (!r_identical(x.value, y.value)) return false;
      } else if (!aval_eq(x.aval, y.aval)) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace rpjrt
