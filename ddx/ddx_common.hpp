/*******************************************************************************
   ddx benchmark wiring shared by the four ddx/*_ddx.cpp kernels: the graph a
   kernel is recorded into, the interpreter row and the 4-lane JIT row.

   Copyright (c) 2026 Sayan Samanta
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#pragma once

#include "jit/kernel.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// What ddx/validate.cpp checks, each a max over BOTH ddx rows (interpreter
// and JIT); negative means "not implemented".
//   gradient_error  max over every input of
//                       |g - g_fd| / (|g_fd| + atol / rtol + 1e-10),
//                   rtol = ddxbench::kGradientTolerance, so passing
//                   (gradient_error < rtol) is the mixed test
//                       |g - g_fd| < rtol |g_fd| + atol.
//                   g_fd is the central difference (f(x+eps) - f(x-eps)) / (2 eps),
//                   eps 1e-6 as in src/fd_benchmarks.cpp.  Not its forward
//                   bump: that one's own error can exceed the tolerance
//                   (1.4e-4 on SABR's alphas, checked against complex step).
//                   atol is g_fd's roundoff floor.  f(x+eps) and f(x-eps)
//                   each carry an error of kappa ulps of |f|, so g_fd is known
//                   only to kappa |f| eps_mach / eps; atol takes kappa =
//                   kFdRoundoffUlps.  A derivative below atol moves f by less
//                   than f's own roundoff: g_fd reads 0 and the test bounds
//                   |g| by atol, which is all a double central difference can
//                   say.  ddxbench::gradient_error() computes this; the
//                   atol = 0 form (heston_ddx.cpp and sabr_calibration_ddx.cpp
//                   as written) is strictly tighter, so their passes hold.
//   value_error     max |f_ddx - f_double| / (|f_double| + 1e-10), f_double
//                   the shared src/*.hpp kernel at the same inputs and samples.
//                   Roundoff-sized: anything larger means a branch-free copy
//                   drifted from the original.
//   graph_build_ms  one cold ddxbench::Graph construction; negative if the
//                   kernel does not measure it.
//   jit_compile_ms  one cold ddxbench::Jit construction (Compiler::create()
//                   plus the LLVM compile); negative if not measured.
struct DdxValidation
{
    double gradient_error = -1.0;
    double value_error = -1.0;
    double graph_build_ms = -1.0;
    double jit_compile_ms = -1.0;
};

DdxValidation ddx_validate_heston();
DdxValidation ddx_validate_sabr();
DdxValidation ddx_validate_xva();
DdxValidation ddx_validate_libor();

namespace ddxbench {

namespace rt = ddx::rt;
namespace jit = ddx::jit;
using RE = rt::RTExpression<double>;

// Matches xad::codegen::CodegenBackendAVX<double>::VECTOR_WIDTH, the width of
// the XAD-Codegen row the ddx-JIT row is compared against.
inline constexpr int kLanes = 4;

// rtol of the gradient check, the tolerance
// xad/samples/LiborSwaptionPricer/benchmark.cpp uses.
inline constexpr double kGradientTolerance = 1e-4;

// kappa in DdxValidation's atol.  XVA-CVA is the worst case: its MC mean
// (6.2e54) is set by a few paths that explode through exp, and against the
// same kernel in long double its double means are off by up to 229 ulps of
// |f| and its g_fd by up to 99.5 |f| eps_mach / eps.  1000 is that bound
// rounded up to the next decade.
inline constexpr double kFdRoundoffUlps = 1000.0;

// DdxValidation::gradient_error for one row: g against g_fd, the central
// difference with step eps of a function whose value is f.
inline double gradient_error(std::span<const double> g, std::span<const double> g_fd,
                             double f, double eps)
{
    const double atol = kFdRoundoffUlps * std::abs(f) *
                        std::numeric_limits<double>::epsilon() / eps;
    const double floor = atol / kGradientTolerance + 1e-10;
    double worst = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i)
        worst = std::max(worst, std::abs(g[i] - g_fd[i]) / (std::abs(g_fd[i]) + floor));
    return worst;
}

// rt::Builder keeps its symbols sorted by name and a symbol's slot is its
// sorted position.  Unpadded, "x10" sorts between "x1" and "x2" and the
// gradient comes back permuted; zero-padded to the width of n-1, name order
// is index order and slot i is input i.
inline std::vector<std::string> padded_names(std::size_t n)
{
    std::vector<std::string> names;
    if (n == 0)
        return names;
    const std::size_t width = std::to_string(n - 1).size();
    names.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::string digits = std::to_string(i);
        names.push_back("x" + std::string(width - digits.size(), '0') + digits);
    }
    return names;
}

// One kernel recorded once: n differentiated inputs (rt::var), m samples
// (rt::seed -- input columns that are never differentiated), the value and
// its Jacobian row.  Pinned in place: Interp and Jit hold a reference to it.
struct Graph
{
    rt::Builder<> b;
    std::size_t n_inputs;
    std::size_t n_seeds;
    rt::NodeId root;
    rt::Jacobian jac;
    rt::Graph<double> graph;

    // f is RE(std::span<const RE> x, std::span<const RE> z).
    template <typename F>
    Graph(std::size_t n, std::size_t m, F&& f)
        : n_inputs(n),
          n_seeds(m),
          root(record(b, n, m, std::forward<F>(f))),
          jac(rt::build_jacobian_impl(b, std::span<const rt::NodeId>{&root, 1})),
          graph(rt::GraphBuilder{b}
                    .values_from(std::span<const rt::NodeId>{&root, 1})
                    .jacobian_from(jac)
                    .finish())
    {
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    // Seed columns the graph kept.  arity() counts seeds up to the highest one
    // still live, so trailing samples the kernel never reads have no column.
    std::size_t live_seeds() const { return graph.arity() - n_inputs; }

private:
    template <typename F>
    static rt::NodeId record(rt::Builder<>& b, std::size_t n, std::size_t m, F&& f)
    {
        const std::vector<std::string> names = padded_names(n);
        std::vector<RE> x;
        x.reserve(n);
        for (const std::string& name : names)
            x.push_back(rt::var(b, name));
        // A throw, not an assert: the harness builds with -DNDEBUG everywhere.
        if (!std::ranges::equal(b.symbols(), names))
            throw std::logic_error("ddx: builder symbols are not in input order");

        std::vector<RE> z;
        z.reserve(m);
        for (std::size_t j = 0; j < m; ++j)
            z.push_back(rt::seed(b, static_cast<std::uint32_t>(j)));

        return f(std::span<const RE>{x}, std::span<const RE>{z}).id(b);
    }
};

// The ddx row: the graph interpreted, one point per sweep.  The point is the
// inputs followed by the seeds, the tape is indexed by node id.
struct Interp
{
    const Graph& g;
    std::vector<double> point;
    std::vector<double> tape;

    explicit Interp(const Graph& graph)
        : g(graph), point(graph.graph.arity()), tape(graph.b.size())
    {
    }

    void set_inputs(std::span<const double> x)
    {
        std::copy_n(x.begin(), g.n_inputs, point.begin());
    }

    // Seeds past live_seeds() are dead in the graph and have no column.
    void set_seeds(std::span<const double> z)
    {
        const std::size_t n = std::min(z.size(), g.live_seeds());
        std::copy_n(z.begin(), n, point.begin() + static_cast<std::ptrdiff_t>(g.n_inputs));
    }

    double run()
    {
        rt::evaluate_into(g.b, point, g.graph.schedule(), std::span<double>{tape});
        return tape[g.root];
    }

    // cell.column is the symbol's sorted index, which padded_names() makes
    // the input index.  Cells the sweep folded to zero are not stored.
    void add_gradient_to(std::span<double> grad) const
    {
        for (const rt::Cell& cell : g.graph.jacobian_pattern().entries())
            grad[cell.column] += tape[g.jac.partial[cell.slot]];
    }
};

// The ddx row over independent paths: the graph interpreted rt::block_lanes
// points per sweep, as rt::Equation interprets a batch -- the dispatch is paid
// once per node per block and each op is a lane loop.  Same surface as Jit.
// point is symbol-major (column i, lane k at i * lanes + k), tape node-major.
struct Block
{
    static constexpr std::size_t lanes = rt::block_lanes;

    const Graph& g;
    std::vector<double> point;
    std::vector<double> tape;

    explicit Block(const Graph& graph)
        : g(graph), point(graph.graph.arity() * lanes), tape(graph.b.size() * lanes)
    {
    }

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    void set_input(std::size_t i, double v)
    {
        std::fill_n(point.begin() + static_cast<std::ptrdiff_t>(i * lanes), lanes, v);
    }

    // Seeds past live_seeds() are dead in the graph and have no column.
    void set_seed_lane(std::size_t j, std::size_t lane, double v)
    {
        if (j < g.live_seeds())
            point[(g.n_inputs + j) * lanes + lane] = v;
    }

    // A short block repeats its last point into the lanes past n_lanes, which
    // are never read back.
    void run(std::size_t n_lanes)
    {
        if (n_lanes == 0 || n_lanes > lanes)
            throw std::out_of_range("ddx: lane count outside the block");
        for (std::size_t j = g.n_inputs; j < g.graph.arity(); ++j)
        {
            double* column = point.data() + j * lanes;
            std::fill(column + n_lanes, column + lanes, column[n_lanes - 1]);
        }
        rt::evaluate_block<lanes>(g.b, point, g.graph.schedule(), std::span<double>{tape});
    }

    double value(std::size_t lane) const { return tape[g.root * lanes + lane]; }

    void add_gradient_to(std::span<double> grad, std::size_t lane) const
    {
        for (const rt::Cell& cell : g.graph.jacobian_pattern().entries())
            grad[cell.column] += tape[g.jac.partial[cell.slot] * lanes + lane];
    }
};

// The ddx-JIT row: the graph compiled at exactly kLanes lanes, kLanes points
// per call, no object cache.  The Compiler is created and the kernel compiled
// in the constructor.  Callers construct it once, outside the timed region,
// and replay it; the compile cost is measured by ad_benchmarks_ddx_validate
// (DdxValidation::jit_compile_ms).
struct Jit
{
    static constexpr std::size_t lanes = kLanes;

    const Graph& g;
    jit::Compiler compiler;
    jit::Kernel kernel;
    std::vector<std::vector<double>> in;       // one column per graph input
    std::vector<std::vector<double>> values;   // the value column
    std::vector<std::vector<double>> partials; // one column per Jacobian slot
    std::vector<const double*> xs;
    std::vector<double*> fs;
    std::vector<double*> gs;

    explicit Jit(const Graph& graph)
        : g(graph),
          compiler(make_compiler()),
          kernel(make_kernel(compiler, graph.graph)),
          in(graph.graph.arity(), std::vector<double>(kLanes)),
          values(graph.graph.layout().values, std::vector<double>(kLanes)),
          partials(graph.graph.layout().jacobian, std::vector<double>(kLanes))
    {
        for (const auto& column : in)
            xs.push_back(column.data());
        for (auto& column : values)
            fs.push_back(column.data());
        for (auto& column : partials)
            gs.push_back(column.data());
    }

    // xs/fs/gs point into the columns.
    Jit(const Jit&) = delete;
    Jit& operator=(const Jit&) = delete;

    void set_input(std::size_t i, double v) { std::ranges::fill(in[i], v); }
    void set_input_lane(std::size_t i, std::size_t lane, double v) { in[i][lane] = v; }

    // Seeds past live_seeds() are dead in the graph and have no column.
    void set_seed_lane(std::size_t j, std::size_t lane, double v)
    {
        if (j < g.live_seeds())
            in[g.n_inputs + j][lane] = v;
    }

    void run(std::size_t n_lanes)
    {
        if (n_lanes > static_cast<std::size_t>(kLanes))
            throw std::out_of_range("ddx: more lanes than the JIT buffers hold");
        kernel(xs, fs, gs, {}, n_lanes);
    }

    double value(std::size_t lane) const { return values[0][lane]; }

    // Same walk as Interp::add_gradient_to, one lane of the partial columns.
    void add_gradient_to(std::span<double> grad, std::size_t lane) const
    {
        for (const rt::Cell& cell : g.graph.jacobian_pattern().entries())
            grad[cell.column] += partials[cell.slot][lane];
    }

private:
    // No silent fallback: a JIT row that quietly interpreted is the one
    // unacceptable outcome.
    static jit::Compiler make_compiler()
    {
        auto c = jit::Compiler::create();
        if (!c)
            throw std::runtime_error("ddx: JIT unavailable: " + c.error().detail);
        return std::move(*c);
    }

    static jit::Kernel make_kernel(jit::Compiler& c, const rt::Graph<double>& graph)
    {
        auto k = c.compile(graph, {.points = kLanes,
                                   .codegen = {.lanes = *jit::Lanes::exactly(kLanes)},
                                   .cache_dir = {}});
        if (!k)
            throw std::runtime_error("ddx: JIT compile failed: " + k.error().detail);
        return std::move(*k);
    }
};

}  // namespace ddxbench
