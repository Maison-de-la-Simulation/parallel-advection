#pragma once

#include <Adv4dParams.hpp>
#include <sycl/sycl.hpp>
#include <array>

/* Lagrange variables, order, number of points, offset from the current point */
int static constexpr LAG_ORDER = 5;
int static constexpr LAG_PTS = 6;
int static constexpr LAG_OFFSET = 2;
real_t static constexpr loc[] = {-1. / 24, 1. / 24.,  -1. / 12.,
                                 1. / 12., -1. / 24., 1. / 24.};

struct LagrangianSolverBase {
    Adv4dParams params_;
    span2d_t efield_;

    LagrangianSolverBase() = delete;
    LagrangianSolverBase(const Adv4dParams &p, span2d_t &efield) :
        params_(p), efield_(efield) {}

    inline constexpr auto window() const { return 1; }

    [[nodiscard]] static inline __attribute__((always_inline)) real_t
    coord(const int i, const real_t &minValue, const real_t &delta) noexcept {
        return minValue + i * delta;
    }

    [[nodiscard]] static inline __attribute__((always_inline))
    std::array<real_t, LAG_PTS> lag_basis(real_t px) noexcept {
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
        coef[4] = loc[4] * pxm2_01 * (pxm2 + 1.) * (5 * sqrpxm2 - 11 * pxm2 - 2.);
        coef[5] = loc[5] * pxm2_01 * pxm2 * (pxm2 + 1.) * (pxm2 - 2.);

        return coef;
    }
};

// =================== VX Solver ===================
struct VxSolver : public LagrangianSolverBase {
    using LagrangianSolverBase::LagrangianSolverBase;

    template <class ArrayLike1D>
    inline __attribute__((always_inline))
    real_t operator()(const ArrayLike1D data, const size_t &i0,
                      const size_t &i1, const size_t &i2) const {
        auto const ix = i0 / params_.ny;
        auto const iy = i0 - ix * params_.ny;
        auto const &ivx = i1;
        auto const &ivy = i2;

        auto const &n1 = params_.nvx;

        const auto vx = coord(ivx, params_.minVx, params_.dvx);
        const auto speed_x = params_.dt * efield_(ix, iy);

        real_t vxFootCoord = sycl::max(params_.minVx, vx - speed_x);

        const int leftNode =
            sycl::floor((vxFootCoord - params_.minVx) * params_.inv_dvx);

        const real_t d_prev1 =
            LAG_OFFSET +
            params_.inv_dvx *
                (vxFootCoord - coord(leftNode, params_.minVx, params_.dvx));

        auto coef = lag_basis(d_prev1);

        const int ipos1 = leftNode - LAG_OFFSET;
        real_t value = 0.;
        for (int k = 0; k <= LAG_ORDER; k++) {
            int id1_ipos = (n1 + ipos1 + k) % n1;
            value += coef[k] * data(id1_ipos);
        }

        return value;
    }
};

// =================== VY Solver ===================
struct VySolver : public LagrangianSolverBase {
    using LagrangianSolverBase::LagrangianSolverBase;

    template <class ArrayLike1D>
    inline __attribute__((always_inline))
    real_t operator()(const ArrayLike1D data, const size_t &i0,
                      const size_t &i1, const size_t &i2) const {
        auto const ix = (i0 / params_.ny) % params_.nx;
        auto const iy = i0 % params_.ny;
        auto const &ivy = i1;

        const auto &n1 = params_.nvy;

        const auto vy = coord(ivy, params_.minVy, params_.dvy);
        const auto speed_y = params_.dt * efield_(ix, iy);

        real_t vyFootCoord = sycl::max(params_.minVy, vy - speed_y);

        const int leftNode =
            sycl::floor((vyFootCoord - params_.minVy) * params_.inv_dvy);

        const real_t d_prev = LAG_OFFSET + params_.inv_dvy *
                (vyFootCoord - coord(leftNode, params_.minVy, params_.dvy));

        auto coef = lag_basis(d_prev);

        const int ipos = leftNode - LAG_OFFSET;
        real_t value = 0.;
        for (int k = 0; k <= LAG_ORDER; k++) {
            int id1_ipos = (n1 + ipos + k) % n1;
            value += coef[k] * data(id1_ipos);
        }

        return value;
    }
};

// =================== X Solver ===================
struct XSolver : public LagrangianSolverBase {
    using LagrangianSolverBase::LagrangianSolverBase;

    template <class ArrayLike1D>
    inline __attribute__((always_inline))
    real_t operator()(const ArrayLike1D data, const size_t &i0,
                      const size_t &i1, const size_t &i2) const {
        auto const ivx = i0 / params_.nvy;
        auto const ivy = i0 % params_.nvy;
        auto const &ix = i1;
        auto const &iy = i2;

        const auto &n1 = params_.nx;

        const auto x = coord(ix, params_.minX, params_.dx);
        const auto speed_x = params_.dt * efield_(ix, iy);

        real_t xFootCoord = sycl::max(params_.minX, x - speed_x);

        const int leftNode =
            sycl::floor((xFootCoord - params_.minX) * params_.inv_dx);

        const real_t d_prev = LAG_OFFSET + params_.inv_dx *
                (xFootCoord - coord(leftNode, params_.minX, params_.dx));

        auto coef = lag_basis(d_prev);

        const int ipos = leftNode - LAG_OFFSET;
        real_t value = 0.;
        for (int k = 0; k <= LAG_ORDER; k++) {
            int id1_ipos = (n1 + ipos + k) % n1;
            value += coef[k] * data(id1_ipos);
        }

        return value;
    }
};

// =================== Y Solver ===================
struct YSolver : public LagrangianSolverBase {
    using LagrangianSolverBase::LagrangianSolverBase;

    template <class ArrayLike1D>
    inline __attribute__((always_inline))
    real_t operator()(const ArrayLike1D data, const size_t &i0,
                      const size_t &i1, const size_t &i2) const {
        const size_t nx = params_.nx;
        const size_t nvy = params_.nvy;
        const size_t nvx = params_.nvx;

        auto tmp = i0;
        auto const ix = tmp % nx;
        tmp /= nx;
        auto const ivy = tmp % nvy;
        auto const ivx = tmp / nvy;

        auto const &iy = i1;

        const auto &n1 = params_.ny;

        const auto y = coord(iy, params_.minY, params_.dy);
        const auto speed_y = params_.dt * efield_(ix, iy);

        real_t yFootCoord = sycl::max(params_.minY, y - speed_y);

        const int leftNode =
            sycl::floor((yFootCoord - params_.minY) * params_.inv_dy);

        const real_t d_prev = LAG_OFFSET + params_.inv_dy *
            (yFootCoord - coord(leftNode, params_.minY, params_.dy));

        auto coef = lag_basis(d_prev);

        const int ipos = leftNode - LAG_OFFSET;
        real_t value = 0.;
        for (int k = 0; k <= LAG_ORDER; k++) {
            int id1_ipos = (n1 + ipos + k) % n1;
            value += coef[k] * data(id1_ipos);
        }

        return value;
    }
};
