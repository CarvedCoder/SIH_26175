// DepthWizard — native kernel unit tests (no framework, assert-style).
// Build & run: cmake --build build && ctest --test-dir build  (or via
// native/build.py --test).

#include "../include/dw_raster.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>

static int failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // ── fill_invalid: single-pixel hole takes the 4-neighbour mean ──
    {
        float d[9] = {1, 2, 3,
                      4, std::nanf(""), 6,
                      7, 8, 9};
        dw::fill_invalid(d, 3, 3);
        CHECK(approx(d[4], 5.0f));
    }

    // ── fill_invalid: last-row / last-column cells still fill ──
    // Regression: the row/col-below sentinel once made the neighbour loop
    // skip entirely on the final row and final column, leaving those holes
    // to the leftover mean.
    {
        float d[9] = {1, 2, 3,
                      4, 5, 6,
                      7, std::nanf(""), std::nanf("")};
        dw::fill_invalid(d, 3, 3);
        for (int i = 0; i < 9; ++i) CHECK(std::isfinite(d[i]));
        CHECK(approx(d[7], (4.f + 5.f + 6.f + 7.f) / 4.f, 1e-4f));
    }

    // ── fill_invalid: finite data is untouched (no iterations needed) ──
    {
        float d[4] = {1, 2, 3, 4};
        long iters = dw::fill_invalid(d, 2, 2);
        CHECK(iters == 0);
        CHECK(approx(d[0], 1.f) && approx(d[3], 4.f));
    }

    // ── fill_invalid: 2x2 NaN block fills from the surrounding ring ──
    {
        float d[16] = {
            1,  2,  3,  4,
            5,  std::nanf(""), std::nanf(""), 8,
            9,  std::nanf(""), std::nanf(""), 12,
            13, 14, 15, 16,
        };
        dw::fill_invalid(d, 4, 4);
        for (int i = 0; i < 16; ++i) CHECK(std::isfinite(d[i]));
        // Every filled value is a neighbour mean, so it must lie within the
        // full data range (the first sweep's means include the top rows).
        CHECK(d[5] >= 1.f && d[5] <= 16.f);
        CHECK(d[10] >= 1.f && d[10] <= 16.f);
    }

    // ── fill_invalid: all-NaN grid falls back without hanging ──
    {
        float d[6] = {std::nanf(""), std::nanf(""), std::nanf(""),
                      std::nanf(""), std::nanf(""), std::nanf("")};
        dw::fill_invalid(d, 2, 3);
        for (int i = 0; i < 6; ++i) CHECK(std::isfinite(d[i]));
    }

    // ── stretch_to_01: affine mapping, clamping, NaN -> 0 ──
    {
        const float in[5] = {-1.f, 0.f, 5.f, 10.f, std::nanf("")};
        float out[5];
        dw::stretch_to_01(in, out, 5, 0.f, 10.f);
        CHECK(approx(out[0], 0.f));    // clamped low
        CHECK(approx(out[1], 0.f));
        CHECK(approx(out[2], 0.5f));
        CHECK(approx(out[3], 1.f));    // clamped high
        CHECK(approx(out[4], 0.f));    // NaN -> 0
    }

    // ── stretch_to_01: degenerate span -> 0 (never divides by ~0) ──
    {
        const float in[2] = {3.f, 3.f};
        float out[2];
        dw::stretch_to_01(in, out, 2, 3.f, 3.f);
        CHECK(approx(out[0], 0.f) && approx(out[1], 0.f));
    }

    // ── quantize_u8: endpoints + rounding ──
    {
        const float in[4] = {0.f, 0.5f, 1.f, 2.f};
        std::uint8_t out[4];
        dw::quantize_u8(in, out, 4);
        CHECK(out[0] == 0);
        CHECK(out[1] == 128);  // 127.5 + 0.5
        CHECK(out[2] == 255);
        CHECK(out[3] == 255);  // clamped
    }

    if (failures == 0) {
        std::printf("dw_raster_tests: all checks passed\n");
        return 0;
    }
    std::printf("dw_raster_tests: %d failure(s)\n", failures);
    return 1;
}
