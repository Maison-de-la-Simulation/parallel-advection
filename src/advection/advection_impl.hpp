#include "i_advection.hpp"

class AdvectorX : IAdvectorX {
  public:
    void operator()(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
                    ADVParams m_params) const;
};

/* Lagrange variables, order, number of points, offset from the current point */
int    static constexpr LAG_ORDER  = 5;
int    static constexpr LAG_PTS    = 6;
int    static constexpr LAG_OFFSET = 2;

// ==========================================
// ==========================================
/* Computes the coefficient for semi lagrangian interp of order 5 */
inline void
lag_basis(const double &px, double* coef){
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
    coef[4] = loc[4] * pxm2_01 * (pxm2 + 1.) * (5 * sqrpxm2 - 11 * pxm2 - 2.);
    coef[5] = loc[5] * pxm2_01 * pxm2 * (pxm2 + 1.) * (pxm2 - 2.);
} // end lag_basis

// ==========================================
// ==========================================
/* Computes the covered distance by x during dt and returns the feet coord */
int
displ(const int &ix, const int &ivx, const ADVParams &params){
    auto const minRealx  = params.minRealx;
    auto const minRealVx = params.minRealVx;
    auto const dx  = params.dx;
    auto const dVx = params.dVx;
    auto const dt  = params.dt;
    auto const realWidthx = params.realWidthx;

    const double x = minRealx  + ix * dx; //real coordinate of particles at ix
    const double vx = minRealVx + ivx * dVx; //real speed of particles at ivx
    const double displx = dt * vx;

    const double xstar =
        minRealx + sycl::fmod(realWidthx + x - displx - minRealx, realWidthx);

   return xstar;
} // end displ