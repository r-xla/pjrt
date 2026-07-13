// The two Engine implementations (see dispatch_engine.h): PjrtEngine, the
// native fast path for PJRTBuffer-backed arrays, and ClosureEngine, the
// generic vehicle for any backend that compiles to an R closure.

#include "dispatch_engine.h"

#include <cstring>
#include <unordered_map>
#include <utility>

#include "buffer.h"
#include "client.h"
#include "device.h"
#include "pjrt_impl.h"
#include "utils.h"

namespace rpjrt {

// ---- The array-wrapper contract -------------------------------------------

// The value of a named field of an AnvlArray leaf, or R_NilValue if absent.
// Only `$data` is guaranteed by anvl's backend contract; an engine reads
// further fields only for a leaf whose backend layout it owns (PjrtEngine, the
// xla backend). Other backends' metadata comes through the extractor instead.
SEXP anvl_field(SEXP leaf, const char* name) {
  SEXP nms = Rf_getAttrib(leaf, R_NamesSymbol);
  if (nms == R_NilValue) return R_NilValue;
  const R_xlen_t n = XLENGTH(leaf);
  for (R_xlen_t k = 0; k < n; ++k) {
    if (!std::strcmp(CHAR(STRING_ELT(nms, k)), name))
      return VECTOR_ELT(leaf, k);
  }
  return R_NilValue;
}

// An array leaf this engine can read: an AnvlArray is a list, and anvl_field()
// reads it with VECTOR_ELT. A value carrying the class but not the type is not
// one an engine can read, so it falls through to the core's documented
// rejection ("expected an AnvlArray, a length-1 atomic scalar, or ...") rather
// than reaching VECTOR_ELT and raising R's low-level type error from the hot
// path.
bool is_anvl_array(SEXP leaf) {
  return TYPEOF(leaf) == VECSXP && Rf_inherits(leaf, "AnvlArray");
}

// The first element of a character field, or "" if it is not a string.
std::string field_string(SEXP v) {
  return (TYPEOF(v) == STRSXP && XLENGTH(v) > 0)
             ? std::string(CHAR(STRING_ELT(v, 0)))
             : std::string();
}

// Whether a logical field is TRUE (absent / NA counts as FALSE).
bool field_true(SEXP v) { return v != R_NilValue && Rf_asLogical(v) == TRUE; }

std::optional<RDataInfo> classify_rdata(SEXP leaf) {
  const SEXPTYPE t = TYPEOF(leaf);
  if (Rf_isObject(leaf)) {
    return std::nullopt;  // classed values are not bare R data
  }
  RDataInfo info;
  switch (t) {
    case REALSXP:
      info.dtype = AnvlDtype::kF32;
      break;
    case INTSXP:
      info.dtype = AnvlDtype::kI32;
      break;
    case LGLSXP:
      info.dtype = AnvlDtype::kBool;
      break;
    default:
      return std::nullopt;
  }
  SEXP dim = Rf_getAttrib(leaf, R_DimSymbol);
  if (dim != R_NilValue) {
    // an array of any rank; NA elements pass through like pjrt_buffer()'s
    const R_xlen_t n = XLENGTH(dim);
    info.shape.reserve(n);
    for (R_xlen_t k = 0; k < n; ++k) info.shape.push_back(INTEGER(dim)[k]);
    return info;
  }
  if (XLENGTH(leaf) != 1)
    return std::nullopt;  // bare vector: not a valid input
  // a scalar literal; reject NA (it has no dtype), keep NaN
  switch (t) {
    case REALSXP:
      if (R_IsNA(REAL(leaf)[0])) return std::nullopt;
      break;
    case INTSXP:
      if (INTEGER(leaf)[0] == NA_INTEGER) return std::nullopt;
      break;
    case LGLSXP:
      if (LOGICAL(leaf)[0] == NA_LOGICAL) return std::nullopt;
      break;
    default:
      break;
  }
  return info;
}

// Falls back to the implicit class of an unclassed value; a matrix reports its
// storage mode rather than "matrix", which is fine -- arrays are valid inputs
// and never reach an error message.
std::string r_class_name(SEXP x) {
  SEXP cls = Rf_getAttrib(x, R_ClassSymbol);
  if (TYPEOF(cls) == STRSXP && XLENGTH(cls) > 0) {
    return std::string(CHAR(STRING_ELT(cls, 0)));
  }
  switch (TYPEOF(x)) {
    case REALSXP:
      return "numeric";
    case INTSXP:
      return "integer";
    case LGLSXP:
      return "logical";
    case STRSXP:
      return "character";
    case CPLXSXP:
      return "complex";
    case RAWSXP:
      return "raw";
    case VECSXP:
      return "list";
    case NILSXP:
      return "NULL";
    case ENVSXP:
      return "environment";
    case CLOSXP:
    case BUILTINSXP:
    case SPECIALSXP:
      return "function";
    default:
      return std::string(Rf_type2char(TYPEOF(x)));
  }
}

std::string leaf_subject(const RTree& in_tree, std::size_t leaf_index) {
  return "input `" + tree_path(in_tree, static_cast<int>(leaf_index) + 1) + "`";
}

// ---- Engine: device canonicalization and the generic aval read -------------

SEXP Engine::canonical_device(SEXP device) {
  // Identity first: an interning backend's device is already canonical, so
  // this is one pointer compare per known device (usually one).
  for (const Rcpp::RObject& c : canonical_devices_) {
    if (SEXP(c) == device) return c;
  }
  // Equality fallback: an equal-but-distinct object maps to the canonical
  // one, so a backend that does not intern still gets one token per device.
  for (const Rcpp::RObject& c : canonical_devices_) {
    if (r_identical(c, device)) return c;
  }
  canonical_devices_.emplace_back(device);
  return canonical_devices_.back();
}

// Reject an aval whose dtype could not be represented: every such leaf would
// share one aval, and two calls on different dtypes would then run each
// other's program. Neither tengen nor pjrt's own dtype table can produce one
// (both are closed over AnvlDtype), so this is a guard, not a path.
static void check_dtype_representable(const aval& a, const RTree& in_tree,
                                      std::size_t leaf_index) {
  if (a.dtype == AnvlDtype::kInvalid) {
    Rcpp::stop("invalid %s: its dtype is not one anvl can represent",
               leaf_subject(in_tree, leaf_index));
  }
}

// Build an aval from a backend extractor's outputs: a tengen DataType object,
// an integer shape, and the ambiguous bit. The values come from the backend's
// accessors (see ClosureEngine::read_array), not from a fixed field layout.
static aval aval_from_tengen(SEXP dtype, SEXP shape, bool ambiguous,
                             const RTree& in_tree, std::size_t leaf_index) {
  if (dtype == R_NilValue || TYPEOF(shape) != INTSXP) {
    Rcpp::stop(
        "invalid %s: the backend extractor must return an aval with a dtype "
        "and an integer shape",
        leaf_subject(in_tree, leaf_index));
  }
  aval a;
  a.dtype = anvl_dtype_from_tengen(dtype);
  a.ambiguous = ambiguous;
  const R_xlen_t nd = XLENGTH(shape);
  a.shape.reserve(nd);
  for (R_xlen_t j = 0; j < nd; ++j) a.shape.push_back(INTEGER(shape)[j]);
  check_dtype_representable(a, in_tree, leaf_index);
  return a;
}

// ---- ClosureEngine ----------------------------------------------------------

// The entry is a compiled R closure. It receives the call's execute-time
// inputs -- array leaves contribute their `$data`, bare R data passes through
// as-is, and statics do not appear at all (they are constants of the compiled
// closure, and a cache hit already proves they match) -- and returns the
// finished, wrapped value. Execution, wrapping, and unwrapping are all the
// backend's business, which is what makes this engine serve a backend pjrt
// knows nothing about.

namespace {

struct ClosureEntry : EntryData {
  explicit ClosureEntry(SEXP fun) : r_fun(fun) {}
  Rcpp::Function r_fun;
};

// Rcpp::Function's own check would reject a non-function with a message that
// does not name `extractor`, so the argument is vetted before it is wrapped.
SEXP require_extractor(SEXP extractor) {
  if (!Rf_isFunction(extractor)) {
    Rcpp::stop("the closure engine requires an `extractor` function");
  }
  return extractor;
}

class ClosureEngine : public Engine {
 public:
  ClosureEngine(std::string backend, SEXP extractor)
      : backend_(std::move(backend)),
        extractor_(require_extractor(extractor)) {}

  // Read metadata through the backend's accessors: extractor(leaf) returns
  // list(aval = list(dtype, shape, ambiguous), device, backend). `$data` is the
  // one field read directly (contract-guaranteed). The aval is built only for a
  // leaf of this dispatcher's backend; a foreign leaf carries its true tag back
  // for the core to reject, and its metadata is neither required nor validated.
  std::optional<ArrayLeaf> read_array(SEXP leaf, const RTree& in_tree,
                                      std::size_t leaf_index) override {
    if (!is_anvl_array(leaf)) return std::nullopt;
    ArrayLeaf al;
    al.data = anvl_field(leaf, "data");
    Rcpp::List meta = extractor_(leaf);
    al.backend = field_string(
        meta.containsElementNamed("backend") ? meta["backend"] : R_NilValue);
    al.device =
        meta.containsElementNamed("device") ? meta["device"] : R_NilValue;
    if (al.backend == backend_) {
      if (!meta.containsElementNamed("aval")) {
        Rcpp::stop("invalid %s: the backend extractor must return an `aval`",
                   leaf_subject(in_tree, leaf_index));
      }
      Rcpp::List av = meta["aval"];
      SEXP dtype = av.containsElementNamed("dtype") ? av["dtype"] : R_NilValue;
      SEXP shape = av.containsElementNamed("shape") ? av["shape"] : R_NilValue;
      const bool amb =
          av.containsElementNamed("ambiguous") && field_true(av["ambiguous"]);
      al.av = aval_from_tengen(dtype, shape, amb, in_tree, leaf_index);
    }
    return al;
  }

  void build_entry(const Rcpp::List& res, CacheEntry& e) const override {
    SEXP r_fun = res.containsElementNamed("r_fun")
                     ? static_cast<SEXP>(res["r_fun"])
                     : R_NilValue;
    if (TYPEOF(r_fun) != CLOSXP) {
      Rcpp::stop(
          "compile callback must return a function `r_fun` "
          "(engine = \"closure\")");
    }
    e.data = std::make_unique<ClosureEntry>(r_fun);
  }

  // Under the pin policy the entry's device is the one the backend compiled
  // `r_fun` for, and `r_fun` is what places the inputs on it -- pjrt cannot,
  // since `$data` is the backend's own array type. So there is nothing to do
  // here either way: the inputs go to the closure exactly as they arrived.
  SEXP run(const CacheEntry& e,
           const std::vector<ExecInput>& exec_inputs) const override {
    const auto* ce = static_cast<const ClosureEntry*>(e.data.get());
    Rcpp::List flat(exec_inputs.size());
    for (std::size_t k = 0; k < exec_inputs.size(); ++k) {
      flat[k] = exec_inputs[k].value;
    }
    return ce->r_fun(flat);
  }

 private:
  std::string backend_;       // the tag this dispatcher's arrays must carry
  Rcpp::Function extractor_;  // reads a leaf's metadata via the backend's
                              // accessors
};

// ---- PjrtEngine -------------------------------------------------------------

// One output-donation phantom buffer to allocate per call (CPU memory mgmt).
struct PhantomSpec {
  PJRT_Buffer_Type dtype = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> shape;
};

// What a compiled PJRT program needs at execute time, plus the material the
// output wrap is built from. Every field is set by build_entry() and read-only
// thereafter, which is what lets run() take the entry by const reference.
//
// A template's slot 0 is `$data` (kTemplateDataSlot): build_templates() puts
// it first, and the per-call wrap writes the output buffer into that slot of
// a shallow copy. Keep the two in sync.
struct PjrtEntry : EntryData {
  PjrtEntry(SEXP exec, SEXP client, SEXP device, SEXP out_tree,
            Rcpp::List templates)
      : exec(exec),
        client(client),
        device(device),
        out_tree(out_tree),
        templates(std::move(templates)) {}

  Rcpp::XPtr<PJRTLoadedExecutable> exec;
  Rcpp::XPtr<PJRTClient> client;            // phantoms, uploads, moves
  Rcpp::XPtr<PJRTDevice> device;            // the entry's device
  Rcpp::XPtr<RTree> out_tree;               // the outputs' structure
  Rcpp::List templates;                     // one template AnvlArray per output
  std::vector<Rcpp::RObject> const_arrays;  // buffers prepended to the inputs
  std::vector<PhantomSpec> phantom_specs;
};

// The `$data` field's position in a wrap template (see PjrtEntry::templates).
constexpr int kTemplateDataSlot = 0;

class PjrtEngine : public Engine {
 public:
  PjrtEngine(std::string backend, bool move_inputs)
      : backend_(std::move(backend)),
        opts_(impl_execution_options_create(std::vector<int64_t>(), 0)),
        move_inputs_(move_inputs) {}

  // Reads the native way: dtype/shape/device all come off the PJRTBuffer in
  // `$data` (it caches them natively and cannot be falsified by a drifted
  // field, which this engine never consults). `$ambiguous` is the only R-list
  // field the aval needs -- the buffer carries no such anvl type-system bit --
  // and `$backend` the only other, for the reject path. The buffer is
  // interpreted only once the leaf's tag matches, so a foreign leaf carries its
  // tag back for the core to reject rather than failing the buffer check here.
  std::optional<ArrayLeaf> read_array(SEXP leaf, const RTree& in_tree,
                                      std::size_t leaf_index) override {
    if (!is_anvl_array(leaf)) return std::nullopt;
    ArrayLeaf al;
    al.data = anvl_field(leaf, "data");
    al.backend = field_string(anvl_field(leaf, "backend"));
    if (al.backend == backend_) {
      if (TYPEOF(al.data) != EXTPTRSXP || !Rf_inherits(al.data, "PJRTBuffer")) {
        Rcpp::stop(
            "invalid %s: an \"%s\" AnvlArray must hold a PJRTBuffer in $data",
            leaf_subject(in_tree, leaf_index), backend_.c_str());
      }
      Rcpp::XPtr<PJRTBuffer> buf(al.data);
      al.av.dtype = anvl_dtype_from_pjrt(buf->element_type());
      al.av.shape = buf->dimensions();
      al.av.ambiguous = field_true(anvl_field(leaf, "ambiguous"));
      check_dtype_representable(al.av, in_tree, leaf_index);
      // Device from the buffer, not $device: interned by PJRT_Device* (see
      // canonical_device) so the token still matches a literal-only call's
      // resolved device, which wraps the same per-client-singleton pointer.
      al.device = device_for_ptr(buf->device_ptr(), buf->get_api());
    }
    return al;
  }

  // A device object is one token per underlying PJRT_Device* (a per-client
  // singleton, stable whether it came from a buffer or from pjrt_device()).
  // Overrides the base's identical()-interning so a buffer-sourced device and a
  // resolver-sourced one collapse to the same canonical object -- and thus the
  // same key token -- letting f(x) and f(1) share an entry on one device.
  SEXP canonical_device(SEXP device) override {
    // Unlike the base implementation, this one dereferences the object. A
    // leaf's device always comes from device_for_ptr() below, but the
    // `default_device` resolver is the backend's own R code and can hand back
    // anything: without this check a foreign external pointer would be
    // reinterpreted as a PJRTDevice, and its garbage PJRT_Device* cached and
    // handed to uploads and execution.
    if (TYPEOF(device) != EXTPTRSXP || !Rf_inherits(device, "PJRTDevice")) {
      Rcpp::stop("the `default_device` resolver must return a PJRTDevice");
    }
    Rcpp::XPtr<PJRTDevice> dev(device);
    return device_for_ptr(dev->device, dev->api);
  }

  void build_entry(const Rcpp::List& res, CacheEntry& e) const override {
    auto named = [&](const char* nm) -> SEXP {
      return res.containsElementNamed(nm) ? static_cast<SEXP>(res[nm])
                                          : R_NilValue;
    };
    auto require_xptr = [&](const char* nm, const char* cls) -> SEXP {
      SEXP v = named(nm);
      if (TYPEOF(v) != EXTPTRSXP || !Rf_inherits(v, cls)) {
        Rcpp::stop("compile callback must return `%s` (a %s)", nm, cls);
      }
      return v;
    };
    SEXP exec = require_xptr("exec", "PJRTLoadedExecutable");
    SEXP client = require_xptr("client", "PJRTClient");
    SEXP device = require_xptr("device", "PJRTDevice");
    SEXP out_tree = require_xptr("out_tree", "RTree");
    SEXP out_avals = named("out_avals");
    if (TYPEOF(out_avals) != VECSXP) {
      Rcpp::stop("compile callback must return `out_avals` (a list)");
    }
    // Both lengths are known here, so a mismatched callback result fails at
    // compile time rather than poisoning the cached entry. The declared count
    // against the executable's real output count can only be checked once it
    // has run (see run()).
    Rcpp::XPtr<RTree> tree(out_tree);
    const int n_tree = tree_size_rec(*tree);
    if (XLENGTH(out_avals) != n_tree) {
      Rcpp::stop("out_avals has length %d but out_tree has %d leaves",
                 static_cast<int>(XLENGTH(out_avals)), n_tree);
    }
    SEXP consts = named("const_arrays");
    if (consts != R_NilValue && TYPEOF(consts) != VECSXP) {
      Rcpp::stop("`const_arrays` must be a list or NULL");
    }
    // Each element is handed to execute as a PJRTBuffer; a wrong-typed
    // external pointer there would be reinterpreted blindly and crash, so
    // check now, like the other xptr fields.
    if (consts != R_NilValue) {
      for (R_xlen_t i = 0; i < XLENGTH(consts); ++i) {
        SEXP c = VECTOR_ELT(consts, i);
        if (TYPEOF(c) != EXTPTRSXP || !Rf_inherits(c, "PJRTBuffer")) {
          Rcpp::stop("`const_arrays[[%d]]` must be a PJRTBuffer",
                     static_cast<int>(i) + 1);
        }
      }
    }

    // The phantom specs are parsed before the templates are built, so a
    // callback that declares a bad dtype is reported against `phantom_specs`
    // (pjrt's own dtype table) rather than against `out_avals` (tengen's).
    std::vector<PhantomSpec> phantom_specs;
    if (res.containsElementNamed("phantom_specs")) {
      Rcpp::List specs = res["phantom_specs"];
      phantom_specs.reserve(specs.size());
      for (R_xlen_t i = 0; i < specs.size(); ++i) {
        Rcpp::List spec = specs[i];
        PhantomSpec ps;
        // Normalize the boolean aliases the R layer also accepts (a tengen
        // BooleanType stringifies as "bool"; pjrt's canonical name is "pred").
        std::string dt = Rcpp::as<std::string>(spec["dtype"]);
        if (dt == "bool" || dt == "i1") dt = "pred";
        ps.dtype = string_to_pjrt_buffer_type(dt);
        ps.shape = Rcpp::as<std::vector<int64_t>>(spec["shape"]);
        phantom_specs.push_back(std::move(ps));
      }
    }

    // The wrap templates, built from the callback's declared output avals --
    // the last thing that can throw.
    Rcpp::List templates = build_templates(out_avals, device);

    auto data = std::make_unique<PjrtEntry>(exec, client, device, out_tree,
                                            std::move(templates));
    data->phantom_specs = std::move(phantom_specs);
    if (consts != R_NilValue) {
      Rcpp::List cl(consts);
      data->const_arrays.reserve(cl.size());
      for (R_xlen_t i = 0; i < cl.size(); ++i) {
        data->const_arrays.emplace_back(cl[i]);
      }
    }
    e.data = std::move(data);
  }

  SEXP run(const CacheEntry& e,
           const std::vector<ExecInput>& exec_inputs) const override {
    const auto* pe = static_cast<const PjrtEntry*>(e.data.get());

    // Assemble the executable's inputs: const_arrays ++ the call's inputs ++
    // freshly allocated phantom donation buffers. A buffer input passes
    // through -- or, under the pin policy, is copied to the entry's device
    // when it lives elsewhere; a bare R literal/array is uploaded to the
    // entry's device (same impls and dtype defaults as pjrt_scalar() /
    // pjrt_buffer()). The GC-rooted `inputs` list is built first and each
    // allocated buffer (copy, upload, phantom) written straight into its
    // slot: it is reachable only through `inputs` (the R GC does not scan C++
    // locals across the next allocation).
    Rcpp::List inputs(pe->const_arrays.size() + exec_inputs.size() +
                      pe->phantom_specs.size());
    R_xlen_t pos = 0;
    for (const Rcpp::RObject& c : pe->const_arrays) inputs[pos++] = c;
    for (const ExecInput& in : exec_inputs) {
      if (!in.upload) {
        if (move_inputs_) {
          Rcpp::XPtr<PJRTBuffer> buf(in.value);
          if (buf->device_ptr() != pe->device->device) {
            // Same plugin <=> same client (clients are per-platform
            // singletons), so a differing API pointer means a cross-client
            // host-roundtrip copy -- mirrors pjrt::copy_buffer().
            const bool cross = buf->get_api().get() != pe->client->api.get();
            inputs[pos++] =
                impl_buffer_copy_to_device(buf, pe->device, pe->client, cross);
            continue;
          }
        }
        inputs[pos++] = in.value;
        continue;
      }
      switch (TYPEOF(in.value)) {
        case REALSXP:
          inputs[pos++] = impl_client_buffer_from_double(
              pe->client, pe->device, in.value, in.av->shape, "f32");
          break;
        case INTSXP:
          inputs[pos++] = impl_client_buffer_from_integer(
              pe->client, pe->device, in.value, in.av->shape, "i32");
          break;
        default:
          inputs[pos++] = impl_client_buffer_from_logical(
              pe->client, pe->device, in.value, in.av->shape, "pred");
          break;
      }
    }
    for (const PhantomSpec& ps : pe->phantom_specs) {
      inputs[pos++] =
          client_buffer_empty(pe->client, pe->device, ps.shape, ps.dtype);
    }

    Rcpp::List out_bufs =
        impl_loaded_executable_execute(pe->exec, inputs, opts_);

    // The declared output count against the real one -- the half of the
    // callback's out_avals claim that only the executable can settle. Cheap,
    // and it keeps a miscounted callback from silently wrapping the wrong
    // buffers.
    const R_xlen_t n_out = out_bufs.size();
    if (n_out != pe->templates.size()) {
      Rcpp::stop(
          "out_tree has %d leaves but the executable returned %d outputs",
          static_cast<int>(pe->templates.size()), static_cast<int>(n_out));
    }

    // Wrap each output: a shallow copy of its template with the buffer
    // written into `$data`, then rebuild the caller's structure from the
    // entry's out_tree. Everything but the buffer is fixed per entry.
    Rcpp::List out_flat(n_out);
    std::vector<SEXP> flat(n_out);
    for (R_xlen_t i = 0; i < n_out; ++i) {
      SEXP w = Rf_shallow_duplicate(VECTOR_ELT(pe->templates, i));
      // No PROTECT needed: no allocation happens between duplicating `w` and
      // storing it into the already-protected `out_flat`, which then roots it.
      SET_VECTOR_ELT(out_flat, i, w);
      SET_VECTOR_ELT(w, kTemplateDataSlot, out_bufs[i]);
      flat[i] = w;
    }
    std::size_t p = 0, li = 0;
    return unflatten_rec(*pe->out_tree, flat, p, li);
  }

 private:
  // One template AnvlArray per output, built on the compile (cold) path from
  // the avals the callback declared: a named list (data = NULL, dtype, shape,
  // device, ambiguous, backend) of class "AnvlArray" -- the wrapper layout an
  // xla leaf carries, which PjrtEngine::read_array reads back as an input. The
  // hot path only shallow-copies a template and drops the output buffer into
  // its `$data` slot.
  //
  // `out_avals[[i]]` is list(dtype = <string>, shape = <integer>, ambiguous =
  // <logical(1)>, the last optional) -- the same shape as the input avals the
  // callback receives in `info$avals`. The dtype name is the canonical one
  // ("f32", "i64", ...); the boolean aliases the R layer also accepts are
  // normalized, as in phantom_specs.
  Rcpp::List build_templates(SEXP out_avals, SEXP device) const {
    const R_xlen_t n_out = XLENGTH(out_avals);
    Rcpp::Environment tengen = Rcpp::Environment::namespace_env("tengen");
    Rcpp::Function as_dtype = tengen["as_dtype"];
    Rcpp::CharacterVector cls = Rcpp::CharacterVector::create("AnvlArray");
    Rcpp::CharacterVector backend = Rcpp::CharacterVector::create(backend_);
    Rcpp::List templates(n_out);
    for (R_xlen_t i = 0; i < n_out; ++i) {
      SEXP av = VECTOR_ELT(out_avals, i);
      if (TYPEOF(av) != VECSXP) {
        Rcpp::stop("`out_avals[[%d]]` must be a list(dtype, shape)",
                   static_cast<int>(i) + 1);
      }
      Rcpp::List aval(av);
      if (!aval.containsElementNamed("dtype") ||
          !aval.containsElementNamed("shape")) {
        Rcpp::stop("`out_avals[[%d]]` must carry $dtype and $shape",
                   static_cast<int>(i) + 1);
      }
      std::string dt = Rcpp::as<std::string>(aval["dtype"]);
      // The wrapper's $dtype is a tengen object, so the name goes to
      // tengen::as_dtype(): canonicalize to AnvlDtype's "bool" (pjrt's own
      // C-API spelling "pred" and the MLIR spelling "i1" are accepted aliases).
      if (dt == "pred" || dt == "i1") dt = "bool";
      Rcpp::IntegerVector shape = Rcpp::as<Rcpp::IntegerVector>(aval["shape"]);
      const bool amb = aval.containsElementNamed("ambiguous") &&
                       Rf_asLogical(aval["ambiguous"]) == TRUE;
      Rcpp::List tmpl = Rcpp::List::create(
          Rcpp::Named("data") = R_NilValue,
          Rcpp::Named("dtype") = as_dtype(Rcpp::CharacterVector::create(dt)),
          Rcpp::Named("shape") = shape, Rcpp::Named("device") = device,
          Rcpp::Named("ambiguous") = amb, Rcpp::Named("backend") = backend);
      tmpl.attr("class") = cls;
      templates[i] = tmpl;
    }
    return templates;
  }

  // The canonical R PJRTDevice for one underlying PJRT_Device*, created on
  // first sight and rooted for the engine's lifetime. Both read_array (from a
  // buffer) and canonical_device (from the resolver) intern through it, so a
  // device resolves to one object -- and one key token -- however it arrived.
  //
  // `api` is by const reference: this runs once per array leaf per call, and
  // the hit path (the common one) never needs a copy of the shared_ptr.
  SEXP device_for_ptr(PJRT_Device* p, const std::shared_ptr<PJRT_Api>& api) {
    auto it = device_cache_.find(p);
    if (it != device_cache_.end()) return it->second;
    auto dev = std::make_unique<PJRTDevice>(p, api);
    Rcpp::XPtr<PJRTDevice> xptr(dev.release(), true);
    xptr.attr("class") = "PJRTDevice";
    return device_cache_.emplace(p, xptr).first->second;
  }

  std::string backend_;
  Rcpp::XPtr<PJRTExecuteOptions> opts_;  // reusable, one per engine
  std::unordered_map<PJRT_Device*, Rcpp::RObject> device_cache_;
  // The pin policy: copy an input to the entry's device when it lives
  // elsewhere. Fixed per dispatcher, so it is state here rather than an
  // argument threaded through every run().
  bool move_inputs_ = false;
};

}  // namespace

std::unique_ptr<Engine> make_engine(const std::string& engine_name,
                                    std::string backend, bool move_inputs,
                                    SEXP extractor) {
  if (engine_name == "pjrt") {
    return std::make_unique<PjrtEngine>(std::move(backend), move_inputs);
  }
  // ClosureEngine needs no device policy of its own (under `move_inputs` the
  // backend's `r_fun` places its own inputs, see its run()), but it does need
  // the extractor to read a leaf's metadata via the backend's accessors.
  if (engine_name == "closure") {
    return std::make_unique<ClosureEngine>(std::move(backend), extractor);
  }
  Rcpp::stop("engine must be \"pjrt\" or \"closure\"");
}

}  // namespace rpjrt
