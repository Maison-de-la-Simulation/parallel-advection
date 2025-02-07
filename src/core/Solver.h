#pragma once

#include "AdvectionParams.h"
#include <sycl/sycl.hpp>

/* Lagrange variables, order, number of points, offset from the current point */
int static constexpr LAG_ORDER = 5;
int static constexpr LAG_PTS = 6;
int static constexpr LAG_OFFSET = 2;
double static constexpr loc[] = {-1. / 24, 1. / 24.,  -1. / 12.,
                                 1. / 12., -1. / 24., 1. / 24.};

// using mdspan1D_stride_t =
//     std::experimental::mdspan<double, std::experimental::dextents<size_t, 1>,
//                               std::experimental::layout_stride>;

// auto slice = std::experimental::submdspan(
//     fdist, i0, std::experimental::full_extent, i2);

struct Solver {
    ADVParams params;

    Solver() = delete;
    Solver(const ADVParams &p) : params(p){};

    // ==========================================
    // ==========================================
    /* Computes the real position of x or speed of vx based on discretization */
    [[nodiscard]] static inline __attribute__((always_inline)) double
    coord(const int i, const double &minValue, const double &delta) noexcept {
        return minValue + i * delta;
    }

    // ==========================================
    // ==========================================
    /* Computes the coefficient for semi lagrangian interp of order 5 */
    [[nodiscard]] static inline
        __attribute__((always_inline)) std::array<double, LAG_PTS>
        lag_basis(double px) noexcept {
        std::array<double, LAG_PTS> coef;

        const double pxm2 = px - 2.;
        const double sqrpxm2 = pxm2 * pxm2;
        const double pxm2_01 = pxm2 * (pxm2 - 1.);

        coef[0] = loc[0] * pxm2_01 * (pxm2 + 1.) * (pxm2 - 2.) * (pxm2 - 1.);
        coef[1] = loc[1] * pxm2_01 * (pxm2 - 2.) * (5 * sqrpxm2 + pxm2 - 8.);
        coef[2] = loc[2] * (pxm2 - 1.) * (pxm2 - 2.) * (pxm2 + 1.) *
                  (5 * sqrpxm2 - 3 * pxm2 - 6.);
        coef[3] = loc[3] * pxm2 * (pxm2 + 1.) * (pxm2 - 2.) *
                  (5 * sqrpxm2 - 7 * pxm2 - 4.);
        coef[4] =
            loc[4] * pxm2_01 * (pxm2 + 1.) * (5 * sqrpxm2 - 11 * pxm2 - 2.);
        coef[5] = loc[5] * pxm2_01 * pxm2 * (pxm2 + 1.) * (pxm2 - 2.);

        return coef;
    }   // end lag_basis

    // ==========================================
    // ==========================================
    /* Computes the covered distance by x during dt. returns the feet coord */
    [[nodiscard]] inline __attribute__((always_inline)) double
    displ(const int i1, const int i0) const noexcept {
        double const x = coord(i1, params.minRealX, params.dx);
        double const vx = coord(i0, params.minRealVx, params.dvx);

        double const displx = params.dt * vx;

        return params.minRealX +
               sycl::fmod(params.realWidthX + x - displx - params.minRealX,
                          params.realWidthX);
    }   // end displ

    // ==========================================
    // ==========================================
    /* The _solve_ function of the algorithm presented */
    template <class ArrayLike1D>
    double operator()(const ArrayLike1D data, const size_t &i0,
                      const size_t &i1, const size_t &i2) const {

        double const xFootCoord = displ(i1, i0);

        // index of the cell to the left of footCoord
        const int leftNode =
            sycl::floor((xFootCoord - params.minRealX) * params.inv_dx);

        const double d_prev1 =
            LAG_OFFSET +
            params.inv_dx *
                (xFootCoord - coord(leftNode, params.minRealX, params.dx));

        auto coef = lag_basis(d_prev1);

        const int ipos1 = leftNode - LAG_OFFSET;

        double value = 0.;
        for (int k = 0; k <= LAG_ORDER; k++) {
            int id1_ipos = (params.n1 + ipos1 + k) % params.n1;

            value += coef[k] * data(id1_ipos);
        }

        return value;
    }
};