#pragma once

#include <AdvectionParams.hpp>
#include <sycl/sycl.hpp>

/* Lagrange variables, order, number of points, offset from the current point */
int static constexpr LAG_ORDER = 5;
int static constexpr LAG_PTS = 6;
int static constexpr LAG_OFFSET = 2;
real_t static constexpr loc[] = {-1. / 24, 1. / 24.,  -1. / 12.,
                                 1. / 12., -1. / 24., 1. / 24.};

struct AdvectionSolver {
    ADVParams params;

    AdvectionSolver() = delete;
    AdvectionSolver(const ADVParams &p) : params(p){};

    auto inline constexpr window() const {return 1;}
    // ==========================================
    // ==========================================
    /* Computes the real position of x or speed of vx based on discretization */
    [[nodiscard]] static inline __attribute__((always_inline)) real_t
    coord(const int i, const real_t &minValue, const real_t &delta) noexcept {
        return minValue + i * delta;
    }

    // ==========================================
    // ==========================================
    /* Computes the coefficient for semi lagrangian interp of order 5 */
    [[nodiscard]] static inline
        __attribute__((always_inline)) std::array<real_t, LAG_PTS>
        lag_basis(real_t px) noexcept {
        std::array<real_t, LAG_PTS> coef;

        const real_t pxm2 = px - 2.;
        const real_t sqrpxm2 = pxm2 * pxm2;
        const real_t pxm2_01 = pxm2 * (pxm2 - 1.);

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
    [[nodiscard]] inline __attribute__((always_inline)) real_t
    displ(const int i1, const int i0) const noexcept {
        real_t const x = coord(i1, params.minRealX, params.dx);
        real_t const vx = coord(i0, params.minRealVx, params.dvx);

        real_t const displx = params.dt * vx;

        return params.minRealX +
               sycl::fmod(params.realWidthX + x - displx - params.minRealX,
                          params.realWidthX);
    }   // end displ

    // ==========================================
    // ==========================================
    /* The _solve_ function of the algorithm presented */
    template <class ArrayLike1D>
    inline __attribute__((always_inline))
    real_t operator()(const ArrayLike1D data, const size_t &i0,
                      const size_t &i1, const size_t &i2) const {

        real_t const xFootCoord = displ(i1, i0);

        // index of the cell to the left of footCoord
        const int leftNode =
            sycl::floor((xFootCoord - params.minRealX) * params.inv_dx);

        const real_t d_prev1 =
            LAG_OFFSET +
            params.inv_dx *
                (xFootCoord - coord(leftNode, params.minRealX, params.dx));

        auto coef = lag_basis(d_prev1);

        const int ipos1 = leftNode - LAG_OFFSET;

        real_t value = 0.;
        for (int k = 0; k <= LAG_ORDER; k++) {
            int id1_ipos = (params.n1 + ipos1 + k) % params.n1;

            value += coef[k] * data(id1_ipos);
        }

        return value;
    }
};