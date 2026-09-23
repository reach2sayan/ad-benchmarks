/*******************************************************************************
   ddx LIBOR Swaption MC benchmark - graph built once and replayed per path;
   interpreter and 4-lane JIT rows.
   Rows: ddx (interpreter); ddx-JIT (compiled, 4 lanes) -- graph build and
   LLVM compile happen once, outside the timed region; their cost is reported
   by ad_benchmarks_ddx_validate.

   161 inputs (delta, lambda[80], L0[80]), lambda.size() / 2 = 40 seeds per
   path.  libor_path_gen takes its samples as doubles and
   libor_value_portfolio branches on T for its payoff, so branch-free copies,
   libor_path_gen_ddx and libor_value_portfolio_ddx, are recorded instead, as
   xad/libor_swaption_jit.cpp records libor_path_gen_jit and
   libor_value_portfolio_jit.

   Copyright (c) 2026 Sayan Samanta
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/libor_swaption.hpp"
#include "ddx_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

namespace {

using ddxbench::RE;

// Input layout: delta at 0, lambda at 1..lambda.size(), L0 after it.  Seed
// layout per path: seed j is the path's j-th sample, j < lambda.size() / 2.

/// libor_path_gen with the samples as graph seeds instead of doubles.
template <class T>
void libor_path_gen_ddx(const T& delta, std::vector<T>& L, const std::vector<T>& lambda,
                        std::span<const T> z)
{
    using std::exp;
    using std::sqrt;

    for (size_t n = 0; n < z.size(); n++)
    {
        T sqez = sqrt(delta) * z[n];
        T v = T(0.0);
        for (size_t i = n + 1; i < L.size(); i++)
        {
            T lam = lambda[i - n - 1];
            T con1 = delta * lam;
            v += (con1 * L[i]) / (T(1.0) + delta * L[i]);
            L[i] *= exp(con1 * v + lam * (sqez - T(0.5) * con1));
        }
    }
}

/// libor_value_portfolio with a branch-free payoff.
template <class T>
T libor_value_portfolio_ddx(const T& delta, const std::vector<int>& maturities,
                            const std::vector<double>& swaprates, const std::vector<T>& L,
                            std::vector<T>& Btmp, std::vector<T>& Stmp)
{
    const size_t NN = L.size();
    const size_t N = NN / 2;
    const size_t Nopt = swaprates.size();

    Btmp.resize(NN);
    Stmp.resize(NN);

    T b = T(1.0);
    T s = T(0.0);

    for (size_t n = N; n < NN; ++n)
    {
        b = b / (T(1.0) + delta * L[n]);
        s = s + delta * b;
        Btmp[n] = b;
        Stmp[n] = s;
    }

    T v = T(0.0);
    for (size_t i = 0; i < Nopt; i++)
    {
        int m = maturities[i] + static_cast<int>(N) - 1;
        T swapval = Btmp[m] + swaprates[i] * Stmp[m] - T(1.0);
        // Payer payoff.  The closed form of the same number as
        // `if (swapval < 0) v += -100 * swapval`, and what XAD-Codegen records
        // via xad::less(swapval, 0).If(-100 * swapval, 0) in
        // xad/libor_swaption_jit.cpp.
        v += T(100.0) * max(-swapval, T(0.0));
    }

    for (size_t n = 0; n < N; n++)
        v = v / (T(1.0) + delta * L[n]);

    return v;
}

auto libor_kernel(const MarketParameters& market, const SwaptionPortfolio& portfolio)
{
    const std::size_t nLambda = market.lambda.size();
    return [&portfolio, nLambda](std::span<const RE> x, std::span<const RE> z)
    {
        std::vector<RE> lambda(x.begin() + 1, x.begin() + 1 + nLambda);
        std::vector<RE> L(x.begin() + 1 + nLambda, x.end());
        std::vector<RE> tmp1, tmp2;
        libor_path_gen_ddx<RE>(x[0], L, lambda, z);
        return libor_value_portfolio_ddx<RE>(x[0], portfolio.maturities,
                                             portfolio.swaprates, L, tmp1, tmp2);
    };
}

std::size_t libor_num_inputs(const MarketParameters& market)
{
    return 1 + market.lambda.size() + market.L0.size();
}

std::vector<double> libor_input_values(const MarketParameters& market)
{
    std::vector<double> vals;
    vals.reserve(libor_num_inputs(market));
    vals.push_back(market.delta);
    vals.insert(vals.end(), market.lambda.begin(), market.lambda.end());
    vals.insert(vals.end(), market.L0.begin(), market.L0.end());
    return vals;
}

MarketParameters libor_market_from(const MarketParameters& market, const double* vals)
{
    MarketParameters m = market;
    m.delta = vals[0];
    std::copy_n(vals + 1, m.lambda.size(), m.lambda.begin());
    std::copy_n(vals + 1 + m.lambda.size(), m.L0.size(), m.L0.begin());
    return m;
}

// The path sums as XAD reports them: symbol 0 is delta, then lambda, then L0,
// each divided by numPaths.
SensitivityResults libor_sensitivities(const MarketParameters& market, double total_price,
                                       const std::vector<double>& total_grad, int numPaths)
{
    const std::size_t nLambda = market.lambda.size();
    SensitivityResults res;
    res.price = total_price / numPaths;
    res.d_delta = total_grad[0] / numPaths;
    res.d_lambda.resize(nLambda);
    res.d_L0.resize(market.L0.size());
    for (std::size_t i = 0; i < nLambda; ++i)
        res.d_lambda[i] = total_grad[1 + i] / numPaths;
    for (std::size_t i = 0; i < market.L0.size(); ++i)
        res.d_L0[i] = total_grad[1 + nLambda + i] / numPaths;
    return res;
}

void libor_sink(const SensitivityResults& res)
{
    volatile double v = res.price;
    (void)v;
    volatile double d = res.d_delta;
    (void)d;
    for (double g : res.d_lambda)
    {
        volatile double gi = g;
        (void)gi;
    }
    for (double g : res.d_L0)
    {
        volatile double gi = g;
        (void)gi;
    }
}

// The path sums of the value and its gradient, before the division by the
// path count.
struct LiborTotals
{
    double price = 0.0;
    std::vector<double> grad;
};

// The gradient sweep over either row, Row::lanes paths per call: the JIT's
// compiled kernel or the interpreter's block sweep.  draw(path) is path's
// samples, asked for in path order and copied before the next call.
template <class Row, class Draw>
LiborTotals libor_sweep(Row& row, const MarketParameters& market, int numPaths,
                        std::size_t numInputs, Draw&& draw)
{
    constexpr int BATCH = static_cast<int>(Row::lanes);

    // delta, lambda, L0 (same for all lanes)
    const std::vector<double> vals = libor_input_values(market);
    for (size_t i = 0; i < numInputs; ++i)
        row.set_input(i, vals[i]);

    LiborTotals t{0.0, std::vector<double>(numInputs, 0.0)};

    // --- Execute the row: BATCH paths per call ---
    const int numBatches = (numPaths + BATCH - 1) / BATCH;
    for (int b = 0; b < numBatches; ++b)
    {
        int batchStart = b * BATCH;
        int actualBatch = std::min(BATCH, numPaths - batchStart);

        // Random samples (different per lane)
        for (int l = 0; l < actualBatch; ++l)
        {
            const std::vector<double>& z = draw(batchStart + l);
            for (size_t m = 0; m < z.size(); ++m)
                row.set_seed_lane(m, l, z[m]);
        }

        row.run(actualBatch);

        for (int l = 0; l < actualBatch; ++l)
        {
            t.price += row.value(l);
            row.add_gradient_to(t.grad, l);
        }
    }
    return t;
}

// Pre-generated samples, one vector per path.
auto libor_drawn(const std::vector<std::vector<double>>& allSamples)
{
    return [&allSamples](int path) -> const std::vector<double>& { return allSamples[path]; };
}

}  // namespace

BenchmarkResult ddx_libor_swaption(int numPaths, size_t warmup, size_t iters)
{
    auto portfolio = defaultPortfolio();
    auto market = defaultMarket();
    const unsigned long long SEED = 12354;
    const size_t numSamples = market.lambda.size() / 2;
    const size_t numInputs = libor_num_inputs(market);

    double primal_ms = benchmark(
        [&]() {
            volatile double v = libor_price_mc(market, portfolio, numPaths, SEED);
            (void)v;
        },
        warmup, iters);

    // Gradient via the interpreted graph, 8 paths per sweep; the samples are
    // drawn inside the timed region, one path at a time
    double grad_ms = benchmark(
        [&]()
        {
            // --- Record the graph (once) ---
            ddxbench::Graph g(numInputs, numSamples, libor_kernel(market, portfolio));
            ddxbench::Block block(g);

            std::mt19937 gen(SEED);
            std::vector<double> samples(numSamples);
            const LiborTotals t = libor_sweep(block, market, numPaths, numInputs,
                                              [&](int) -> const std::vector<double>& {
                                                  generateSamples(gen, samples);
                                                  return samples;
                                              });

            libor_sink(libor_sensitivities(market, t.price, t.grad, numPaths));
        },
        warmup, iters);

    return {"ddx", "LiborSwaption", primal_ms, grad_ms};
}

BenchmarkResult ddx_jit_libor_swaption(int numPaths, size_t warmup, size_t iters)
{
    auto portfolio = defaultPortfolio();
    auto market = defaultMarket();
    const unsigned long long SEED = 12354;
    const size_t numSamples = market.lambda.size() / 2;
    const size_t numInputs = libor_num_inputs(market);

    auto allSamples = libor_generate_all_samples(numPaths, numSamples, SEED);

    double primal_ms = benchmark(
        [&]() {
            volatile double v = libor_price_mc(market, portfolio, numPaths, SEED);
            (void)v;
        },
        warmup, iters);

    // --- Record and compile the graph (once, untimed) ---
    ddxbench::Graph g(numInputs, numSamples, libor_kernel(market, portfolio));
    ddxbench::Jit jit(g);

    // Gradient via the JIT-compiled graph, 4 paths per call
    double grad_ms = benchmark(
        [&]()
        {
            const LiborTotals t =
                libor_sweep(jit, market, numPaths, numInputs, libor_drawn(allSamples));
            libor_sink(libor_sensitivities(market, t.price, t.grad, numPaths));
        },
        warmup, iters);

    return {"ddx-JIT", "LiborSwaption", primal_ms, grad_ms};
}

// The MC mean over 1000 paths at the default market, both rows, against
// central differences of the libor_path_gen + libor_value_portfolio double MC
// mean and against that mean itself, as DdxValidation describes.  Every
// evaluation uses the same pre-generated samples.  The reference is 2 * 161
// bumped MC runs.
DdxValidation ddx_validate_libor()
{
    constexpr double eps = 1e-6;
    constexpr int numPaths = 1000;

    auto portfolio = defaultPortfolio();
    auto market = defaultMarket();
    const unsigned long long SEED = 12354;
    const size_t numSamples = market.lambda.size() / 2;
    const size_t numInputs = libor_num_inputs(market);

    auto allSamples = libor_generate_all_samples(numPaths, numSamples, SEED);
    const std::vector<double> vals = libor_input_values(market);

    auto mc_mean = [&](const double* x)
    {
        const MarketParameters m = libor_market_from(market, x);
        std::vector<double> L, tmp1, tmp2;
        double sum = 0.0;
        for (int p = 0; p < numPaths; ++p)
        {
            L.assign(m.L0.begin(), m.L0.end());
            libor_path_gen(m.delta, L, m.lambda, allSamples[p]);
            sum += libor_value_portfolio(m.delta, portfolio.maturities, portfolio.swaprates,
                                         L, tmp1, tmp2);
        }
        return sum / numPaths;
    };

    const double f_double = mc_mean(vals.data());
    std::vector<double> g_fd(numInputs);
    std::vector<double> bumped = vals;
    for (size_t j = 0; j < numInputs; ++j)
    {
        bumped[j] = vals[j] + eps;
        const double f_up = mc_mean(bumped.data());
        bumped[j] = vals[j] - eps;
        const double f_down = mc_mean(bumped.data());
        bumped[j] = vals[j];
        g_fd[j] = (f_up - f_down) / (2.0 * eps);
    }

    DdxValidation r{0.0, 0.0};
    auto check = [&](double f, const std::vector<double>& grad)
    {
        r.value_error = std::max(r.value_error,
                                 std::abs(f - f_double) / (std::abs(f_double) + 1e-10));
        r.gradient_error = std::max(r.gradient_error,
                                    ddxbench::gradient_error(grad, g_fd, f_double, eps));
    };

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    const auto graph_start = Clock::now();
    ddxbench::Graph g(numInputs, numSamples, libor_kernel(market, portfolio));
    r.graph_build_ms = Ms(Clock::now() - graph_start).count();

    auto check_totals = [&](LiborTotals t)
    {
        for (double& gj : t.grad)
            gj /= numPaths;
        check(t.price / numPaths, t.grad);
    };

    {
        ddxbench::Block block(g);
        check_totals(libor_sweep(block, market, numPaths, numInputs, libor_drawn(allSamples)));
    }

    {
        const auto jit_start = Clock::now();
        ddxbench::Jit jit(g);
        r.jit_compile_ms = Ms(Clock::now() - jit_start).count();
        check_totals(libor_sweep(jit, market, numPaths, numInputs, libor_drawn(allSamples)));
    }

    return r;
}
