#pragma once

#include "AdvectionParams.h"
#include <sycl/sycl.hpp>

/* Lagrange variables, order, number of points, offset from the current point */
int    static constexpr LAG_ORDER  = 5;
int    static constexpr LAG_PTS    = 6;
int    static constexpr LAG_OFFSET = 2;

class IAdvectorX {
public:
    virtual ~IAdvectorX() = default;

    virtual sycl::event operator()(
        sycl::queue &Q,
        sycl::buffer<double, 2> &buff_fdistrib,
        const ADVParams &params) const = 0;

    // ==========================================
    // ==========================================
    /* Computes the coefficient for semi lagrangian interp of order 5 */
    inline void lag_basis(const double &px, double *coef) const {
        constexpr double loc[] = {-1. / 24, 1. / 24.,  -1. / 12.,
                                  1. / 12., -1. / 24., 1. / 24.};
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
    }   // end lag_basis

    // ==========================================
    // ==========================================
    /* Computes the covered distance by x during dt and returns the feet coord
     */
    inline double displ(const int &ix, const int &ivx,
                     const ADVParams &params) const {
        auto const minRealx = params.minRealx;
        auto const minRealVx = params.minRealVx;
        auto const dx = params.dx;
        auto const dVx = params.dVx;
        auto const dt = params.dt;
        auto const realWidthx = params.realWidthx;

        double const x =
            minRealx + ix * dx;   // real coordinate of particles at ix
        double const vx =
            minRealVx + ivx * dVx;   // real speed of particles at ivx
        double const displx = dt * vx;

        double const xstar =
            minRealx +
            sycl::fmod(realWidthx + x - displx - minRealx, realWidthx);

        // if(ivx ==0){
        //     std::cout << "xstar - x:";
        //     std::cout << xstar - x << std::endl;

        // }

        return xstar;
    }   // end displ
};