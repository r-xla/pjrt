// The two Engine implementations (see dispatch_engine.h): PjrtEngine, the
// native fast path for PJRTBuffer-backed arrays, and ClosureEngine, the
// generic vehicle for any backend that compiles to an R closure.

#include "dispatch_engine.h"

#include <cstring>
#include <utility>

#include "buffer.h"
#include "client.h"
#include "device.h"
#include "pjrt_impl.h"
#include "utils.h"

namespace rpjrt {

// ---- The array-wrapper contract -------------------------------------------

std::optional<AnvlFields> anvl_fields(SEXP leaf) {
  if (TYPEOF(leaf) != VECSXP || !Rf_inherits(leaf, "AnvlArray")) {
    return std::nullopt;
  }
  SEXP nms = Rf_getAttrib(leaf, R_NamesSymbol);
  if (nms == R_NilValue) return std::nullopt;
  AnvlFields f;
  SEXP amb = R_NilValue;
  for (R_xlen_t k = 0; k < XLENGTH(leaf); ++k) {
    const char* nm = CHAR(STRING_ELT(nms, k));
    if (!std::strcmp(nm, "data"))
      f.data = VECTOR_ELT(leaf, k);
    else if (!std::strcmp(nm, "backend"))
      f.backend = TYPEOF(VECTOR_ELT(leaf, k)) == STRSXP
                      ? CHAR(STRING_ELT(VECTOR_ELT(leaf, k), 0))
                      : nullptr;
    else if (!std::strcmp(nm, "ambiguous"))
      amb = VECTOR_ELT(leaf, k);
    else if (!std::strcmp(nm, "dtype"))
      f.dtype = VECTOR_ELT(leaf, k);
    else if (!std::strcmp(nm, "shape"))
      f.shape = VECTOR_ELT(leaf, k);
    else if (!std::strcmp(nm, "device"))
      f.device = VECTOR_ELT(leaf, k);
  }
  f.ambiguous = (amb != R_NilValue && Rf_asLogical(amb) == TRUE);
  return f;
}

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

Engine::~Engine() {
  for (SEXP c : canonical_devices_) R_ReleaseObject(c);
}

SEXP Engine::canonical_device(SEXP device) {
  // Identity first: an interning backend's device is already canonical, so
  // this is one pointer compare per known device (usually one).
  for (SEXP c : canonical_devices_) {
    if (c == device) return c;
  }
  // Equality fallback: an equal-but-distinct object maps to the canonical
  // one, so a backend that does not intern still gets one token per device.
  for (SEXP c : canonical_devices_) {
    if (r_identical(c, device)) return c;
  }
  R_PreserveObject(device);
  canonical_devices_.push_back(device);
  return device;
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

aval Engine::array_aval(const AnvlFields& af, const RTree& in_tree,
                        std::size_t leaf_index) const {
  if (af.dtype == R_NilValue || TYPEOF(af.shape) != INTSXP) {
    Rcpp::stop(
        "invalid %s: an AnvlArray must carry $dtype and an integer "
        "$shape",
        leaf_subject(in_tree, leaf_index));
  }
  aval a;
  a.dtype = anvl_dtype_from_tengen(af.dtype);
  a.ambiguous = af.ambiguous;
  const R_xlen_t nd = XLENGTH(af.shape);
  a.shape.reserve(nd);
  for (R_xlen_t j = 0; j < nd; ++j) {
    a.shape.push_back(INTEGER(af.shape)[j]);
  }
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
  SEXP r_fun = R_NilValue;
};

class ClosureEngine : public Engine {
 public:
  void build_entry(const Rcpp::List& res, CacheEntry& e) const override {
    SEXP r_fun = res.containsElementNamed("r_fun")
                     ? static_cast<SEXP>(res["r_fun"])
                     : R_NilValue;
    if (TYPEOF(r_fun) != CLOSXP) {
      Rcpp::stop(
          "compile callback must return a function `r_fun` "
          "(engine = \"closure\")");
    }
    auto data = std::make_unique<ClosureEntry>();
    data->r_fun = e.preserve(r_fun);
    e.data = std::move(data);
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
    Rcpp::Function fun(ce->r_fun);
    return fun(flat);
  }
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
  SEXP exec = R_NilValue;          // PJRTLoadedExecutable xptr
  std::vector<SEXP> const_arrays;  // buffers prepended to the inputs
  std::vector<PhantomSpec> phantom_specs;
  SEXP client = R_NilValue;     // PJRTClient xptr (phantoms, uploads, moves)
  SEXP device = R_NilValue;     // PJRTDevice xptr: the entry's device
  SEXP out_tree = R_NilValue;   // RTree xptr: the outputs' structure
  SEXP templates = R_NilValue;  // VECSXP: one template AnvlArray per output
};

// The `$data` field's position in a wrap template (see PjrtEntry::templates).
constexpr int kTemplateDataSlot = 0;

class PjrtEngine : public Engine {
 public:
  PjrtEngine(std::string backend, bool move_inputs)
      : backend_(std::move(backend)), move_inputs_(move_inputs) {
    Rcpp::XPtr<PJRTExecuteOptions> opts =
        impl_execution_options_create(std::vector<int64_t>(), 0);
    opts_ = opts;
    R_PreserveObject(opts_);
  }

  ~PjrtEngine() override { R_ReleaseObject(opts_); }

  // The fast aval read: `$data` is a PJRTBuffer that caches its element type
  // and dimensions natively, so the tengen `$dtype` object is never decoded
  // on the hot path. The generic read and this one agree by construction; the
  // buffer is merely cheaper, and it cannot be falsified by a `$dtype` field
  // that drifted.
  aval array_aval(const AnvlFields& af, const RTree& in_tree,
                  std::size_t leaf_index) const override {
    if (TYPEOF(af.data) != EXTPTRSXP || !Rf_inherits(af.data, "PJRTBuffer")) {
      Rcpp::stop(
          "invalid %s: an \"%s\" AnvlArray must hold a PJRTBuffer in $data",
          leaf_subject(in_tree, leaf_index), backend_.c_str());
    }
    Rcpp::XPtr<PJRTBuffer> buf(af.data);
    aval a;
    a.dtype = anvl_dtype_from_pjrt(buf->element_type());
    a.shape = buf->dimensions();
    a.ambiguous = af.ambiguous;
    check_dtype_representable(a, in_tree, leaf_index);
    return a;
  }

  void build_entry(const Rcpp::List& res, CacheEntry& e) const override {
    // Extract everything that can throw FIRST (while `res` keeps the SEXPs
    // rooted), and only then preserve into the entry -- a malformed callback
    // result must not leak a half-preserved entry.
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

    auto data = std::make_unique<PjrtEntry>();
    if (res.containsElementNamed("phantom_specs")) {
      Rcpp::List specs = res["phantom_specs"];
      data->phantom_specs.reserve(specs.size());
      for (R_xlen_t i = 0; i < specs.size(); ++i) {
        Rcpp::List spec = specs[i];
        PhantomSpec ps;
        // Normalize the boolean aliases the R layer also accepts (a tengen
        // BooleanType stringifies as "bool"; pjrt's canonical name is "pred").
        std::string dt = Rcpp::as<std::string>(spec["dtype"]);
        if (dt == "bool" || dt == "i1") dt = "pred";
        ps.dtype = string_to_pjrt_buffer_type(dt);
        ps.shape = Rcpp::as<std::vector<int64_t>>(spec["shape"]);
        data->phantom_specs.push_back(std::move(ps));
      }
    }

    // The wrap templates, built from the callback's declared output avals --
    // the last thing that can throw. Rcpp roots `templates` meanwhile.
    Rcpp::List templates = build_templates(out_avals, device);

    // No throwing operations past this point: preserve the R objects.
    data->exec = e.preserve(exec);
    data->client = e.preserve(client);
    data->device = e.preserve(device);
    data->out_tree = e.preserve(out_tree);
    data->templates = e.preserve(templates);
    if (consts != R_NilValue) {
      Rcpp::List cl(consts);
      data->const_arrays.reserve(cl.size());
      for (R_xlen_t i = 0; i < cl.size(); ++i) {
        data->const_arrays.push_back(e.preserve(cl[i]));
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
    for (SEXP c : pe->const_arrays) inputs[pos++] = c;
    for (const ExecInput& in : exec_inputs) {
      if (!in.upload) {
        if (move_inputs_) {
          Rcpp::XPtr<PJRTBuffer> buf(in.value);
          Rcpp::XPtr<PJRTDevice> dev(pe->device);
          if (buf->device_ptr() != dev->device) {
            Rcpp::XPtr<PJRTClient> client(pe->client);
            // Same plugin <=> same client (clients are per-platform
            // singletons), so a differing API pointer means a cross-client
            // host-roundtrip copy -- mirrors pjrt::copy_buffer().
            const bool cross = buf->get_api().get() != client->api.get();
            inputs[pos++] = impl_buffer_copy_to_device(buf, dev, client, cross);
            continue;
          }
        }
        inputs[pos++] = in.value;
        continue;
      }
      Rcpp::XPtr<PJRTClient> client(pe->client);
      Rcpp::XPtr<PJRTDevice> dev(pe->device);
      switch (TYPEOF(in.value)) {
        case REALSXP:
          inputs[pos++] = impl_client_buffer_from_double(client, dev, in.value,
                                                         in.av->shape, "f32");
          break;
        case INTSXP:
          inputs[pos++] = impl_client_buffer_from_integer(client, dev, in.value,
                                                          in.av->shape, "i32");
          break;
        default:
          inputs[pos++] = impl_client_buffer_from_logical(client, dev, in.value,
                                                          in.av->shape, "pred");
          break;
      }
    }
    if (!pe->phantom_specs.empty()) {
      Rcpp::XPtr<PJRTClient> client(pe->client);
      Rcpp::XPtr<PJRTDevice> device(pe->device);
      for (const PhantomSpec& ps : pe->phantom_specs) {
        inputs[pos++] = client_buffer_empty(client, device, ps.shape, ps.dtype);
      }
    }

    Rcpp::XPtr<PJRTLoadedExecutable> exec(pe->exec);
    Rcpp::XPtr<PJRTExecuteOptions> opts(opts_);
    Rcpp::List out_bufs = impl_loaded_executable_execute(exec, inputs, opts);

    // The declared output count against the real one -- the half of the
    // callback's out_avals claim that only the executable can settle. Cheap,
    // and it keeps a miscounted callback from silently wrapping the wrong
    // buffers.
    const R_xlen_t n_out = out_bufs.size();
    if (n_out != XLENGTH(pe->templates)) {
      Rcpp::stop(
          "out_tree has %d leaves but the executable returned %d outputs",
          static_cast<int>(XLENGTH(pe->templates)), static_cast<int>(n_out));
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
    Rcpp::XPtr<RTree> tree(pe->out_tree);
    std::size_t p = 0, li = 0;
    return unflatten_rec(*tree, flat, p, li);
  }

 private:
  // One template AnvlArray per output, built on the compile (cold) path from
  // the avals the callback declared: a named list (data = NULL, dtype, shape,
  // device, ambiguous, backend) of class "AnvlArray" -- the same field
  // contract anvl_fields() reads. The hot path only shallow-copies a template
  // and drops the output buffer into its `$data` slot.
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

  std::string backend_;
  SEXP opts_ = R_NilValue;  // reusable PJRTExecuteOptions xptr
  // The pin policy: copy an input to the entry's device when it lives
  // elsewhere. Fixed per dispatcher, so it is state here rather than an
  // argument threaded through every run().
  bool move_inputs_ = false;
};

}  // namespace

std::unique_ptr<Engine> make_engine(const std::string& engine_name,
                                    std::string backend, bool move_inputs) {
  if (engine_name == "pjrt") {
    return std::make_unique<PjrtEngine>(std::move(backend), move_inputs);
  }
  // ClosureEngine needs no policy of its own: under `move_inputs` the backend's
  // `r_fun` places its own inputs (see its run()).
  if (engine_name == "closure") {
    return std::make_unique<ClosureEngine>();
  }
  Rcpp::stop("engine must be \"pjrt\" or \"closure\"");
}

}  // namespace rpjrt
