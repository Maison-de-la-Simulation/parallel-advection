#include "vx_advectors.h"

void
advector::vx::MDRange::operator()(KV_double_3d &fdist, KV_double_1d &elec_field,
                                  const ADVParams &params) noexcept {

    // auto const nx = params.nx;
    auto const nvx = params.nvx;
    // auto const n_fict = params.n_fict_dim;
    auto const minRealVx = params.minRealVx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const inv_dvx = params.inv_dvx;
    auto const realWidthVx = params.realWidthVx;

    const Kokkos::Array<int, 3> begin{0, 0, 0};
    const Kokkos::Array<int, 3> end{fdist.extent_int(0), fdist.extent_int(1),
                                    fdist.extent_int(2)};

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy(begin, end);

    Kokkos::parallel_for(
        "MDrange_advectionVx", mdrange_policy,
        KOKKOS_CLASS_LAMBDA(int i, int j, int k) {
            const auto i_fict = i;
            const auto ivx = j;
            const auto ix = k;

            auto const ex = elec_field(ix);
            auto const displx = dt * ex;
            double const vx = coord(ivx, minRealVx, dvx);

            auto const vxFootCoord =
                minRealVx + Kokkos::fmod(realWidthVx + vx - displx - minRealVx,
                                         realWidthVx);

            const int LeftDiscreteNode =
                Kokkos::floor((vxFootCoord - minRealVx) * inv_dvx);

            const double d_prev1 =
                LAG_OFFSET + inv_dvx * (vxFootCoord - coord(LeftDiscreteNode,
                                                            minRealVx, dvx));

            auto coef = lag_basis(d_prev1);

            const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

            m_ftmp(i_fict, ivx, ix) = 0;
            for (int k = 0; k <= LAG_ORDER; k++) {
                int idx_coef = (nvx + ipos1 + k) % nvx;

                m_ftmp(i_fict, ivx, ix) +=
                    coef[k] * fdist(i_fict, idx_coef, ix);
            }
        });

    Kokkos::deep_copy(fdist, m_ftmp);
}