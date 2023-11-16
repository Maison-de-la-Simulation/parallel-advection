#pragma once

#include "AdvectionParams.h"
#include <sycl/sycl.hpp>

/* Lagrange variables, order, number of points, offset from the current point */
int static constexpr LAG_ORDER = 5;
int static constexpr LAG_PTS = 6;
int static constexpr LAG_OFFSET = 2;
double static constexpr loc[] = {-1. / 24, 1. / 24.,  -1. / 12.,
                                 1. / 12., -1. / 24., 1. / 24.};

class IAdvectorX {
  public:
    virtual ~IAdvectorX() = default;

    virtual sycl::event operator()(sycl::queue &Q,
                                   sycl::buffer<double, 2> &buff_fdistrib,
                                   const ADVParams &params) noexcept = 0;

    // ==========================================
    // ==========================================
    /* Computes the real position of x or speed of vx based on discretization */
    [[nodiscard]] static inline __attribute__((always_inline)) double
    coord(const int i, const double &minValue,
          const double &delta) noexcept {
        return minValue + i * delta;
    }

    // ==========================================
    // ==========================================
    /* Computes the coefficient for semi lagrangian interp of order 5 */
    [[nodiscard]] static inline __attribute__((always_inline))
    std::array<double, LAG_PTS>
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
    [[nodiscard]] static inline __attribute__((always_inline)) double
    displ(const int ix, const int ivx, const ADVParams &params) noexcept {
        auto const minRealX = params.minRealX;
        auto const minRealVx = params.minRealVx;
        auto const dx = params.dx;
        auto const dvx = params.dvx;
        auto const dt = params.dt;
        auto const realWidthX = params.realWidthX;

        double const x = coord(ix, minRealX, dx);
        double const vx = coord(ivx, minRealVx, dvx);

        double const displx = dt * vx;

        return minRealX +
               sycl::fmod(realWidthX + x - displx - minRealX, realWidthX);
    }   // end displ

    // ==========================================
    // ==========================================
    [[nodiscard]] static inline __attribute__((always_inline)) double
    displ(const int ix,
          const int ivx,
          const double minRealX,
          const double minRealVx,
          const double realWidthX,
          const double dx,
          const double dvx,
          const double dt
          ) noexcept {

        double const x = coord(ix, minRealX, dx);
        double const vx = coord(ivx, minRealVx, dvx);

        double const displx = dt * vx;

        return minRealX +
               sycl::fmod(realWidthX + x - displx - minRealX, realWidthX);
    }   // end displ
};
