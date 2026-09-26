// DepthWizard — native raster kernels (implementation)
//
// Written for dense, branch-lean loops so the compiler can vectorize.
// The fill kernel keeps a single pass per iteration (one NaN check per
// sample, 8 neighbour reads) instead of the NumPy version's 8 full-array
// shifted copies + reductions per iteration — that allocation traffic was
// the measured bottleneck, not the arithmetic.

#include "dw_raster.hpp"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace dw {

long fill_invalid(float* data, long rows, long cols) {
    const long n = rows * cols;
    if (n <= 0) return 0;

    // Pass 0: collect invalid indices + mean of finite samples (the final
    // fill for unreachable holes). float32 accumulation matches np.mean /
    // the Python fallback's elementwise sum rounding.
    std::vector<long> invalid;
    float finite_sum = 0.0f;
    long finite_count = 0;
    for (long i = 0; i < n; ++i) {
        const float v = data[i];
        if (std::isfinite(v)) {
            finite_sum += v;
            ++finite_count;
        } else {
            invalid.push_back(i);
        }
    }
    if (invalid.empty()) return 0;
    if (finite_count == 0) {
        // All-invalid grid: the Python fallback fills 0.0 — match it.
        for (long i = 0; i < n; ++i) data[i] = 0.0f;
        return 0;
    }

    // Jacobi sweeps over ONLY the still-invalid cells. Every sweep reads
    // the pre-sweep state (reads happen before any write — writes are
    // deferred to the end of the sweep), which is numerically identical to
    // the pure-Python implementation: same update order per pixel, same
    // convergence, same values. Work shrinks with the hole area instead of
    // costing full-array copies per iteration (the naive double-buffer
    // version measured NO speedup over NumPy — the workload is dominated
    // by memory traffic, so eliminating it is what wins).
    std::vector<long> still_invalid;
    std::vector<std::pair<long, float>> updates;
    long iterations = 0;
    for (;;) {
        ++iterations;
        still_invalid.clear();
        updates.clear();
        for (long idx : invalid) {
            const long r = idx / cols;
            const long c = idx - r * cols;
            const long r_up = (r > 0) ? r - 1 : 0;
            const long r_dn = (r + 1 < rows) ? r + 1 : rows - 1;
            float sum = 0.0f;
            int count = 0;
            for (long rr = r_up; rr <= r_dn; ++rr) {
                const long c_l = (c > 0) ? c - 1 : 0;
                const long c_r = (c + 1 < cols) ? c + 1 : cols - 1;
                for (long cc = c_l; cc <= c_r; ++cc) {
                    if (rr == r && cc == c) continue;
                    const float nv = data[rr * cols + cc];
                    if (std::isfinite(nv)) {
                        sum += nv;
                        ++count;
                    }
                }
            }
            if (count > 0) {
                updates.emplace_back(idx, sum / static_cast<float>(count));
            } else {
                still_invalid.push_back(idx);
            }
        }
        // Deferred apply = Jacobi (all reads saw the pre-sweep state).
        for (const auto& [idx, value] : updates) data[idx] = value;
        invalid.swap(still_invalid);
        if (invalid.empty()) break;
        // Hard safety cap: each iteration fills at least the first ring of
        // every hole, so iterations <= max hole radius + 1. 64 matches the
        // NumPy implementation's cap. Leftovers take the mean of the
        // FINITE values at cap time — the Python fallback computes its
        // leftover fill from the post-iteration state, not pass 0.
        if (iterations >= 64) {
            float leftover_sum = 0.0f;
            long leftover_finite = 0;
            for (long i = 0; i < n; ++i) {
                if (std::isfinite(data[i])) {
                    leftover_sum += data[i];
                    ++leftover_finite;
                }
            }
            const float leftover_mean = leftover_finite > 0
                ? leftover_sum / static_cast<float>(leftover_finite)
                : 0.0f;
            for (long idx : invalid) data[idx] = leftover_mean;
            break;
        }
    }
    return iterations;
}

void stretch_to_01(const float* in, float* out, long count, float lo, float hi) {
    const float span = hi - lo;
    const float inv = (span > 1e-9f || span < -1e-9f) ? 1.0f / span : 0.0f;
    for (long i = 0; i < count; ++i) {
        const float v = in[i];
        if (!std::isfinite(v)) {
            out[i] = 0.0f;
            continue;
        }
        float t = (v - lo) * inv;
        // Branchless clamp — vectorizes to minps/maxps.
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        out[i] = t;
    }
}

void quantize_u8(const float* in, std::uint8_t* out, long count) {
    for (long i = 0; i < count; ++i) {
        float t = in[i];
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        // x255 + 0.5 then truncate == round-half-up, matching NumPy's
        // .round() closely enough for display data (<=1 LSB difference on
        // exact .5 ties only).
        out[i] = static_cast<std::uint8_t>(t * 255.0f + 0.5f);
    }
}

}  // namespace dw
