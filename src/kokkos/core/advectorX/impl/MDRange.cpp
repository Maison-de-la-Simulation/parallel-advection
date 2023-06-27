#include "x_advectors.h"

void
advector::x::MDRange::operator()(KV_double_3d &fdist,
                                 const ADVParams &params) noexcept {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const Kokkos::Array<int, 3> begin{0, 0, 0};
    // const Kokkos::Array<int, 3> end{2, 2, 2};
    const Kokkos::Array<int, 3> end{fdist.extent_int(0), fdist.extent_int(1),
                                    fdist.extent_int(2)};

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy(begin, end);

    Kokkos::parallel_for(
        "MDrange_advection", mdrange_policy,
        KOKKOS_CLASS_LAMBDA(int i, int j, int k) {
            const auto i_fict = i;
            const auto ivx = j;
            const auto ix = k;

            double const xFootCoord = displ(ix, ivx, params);

            const double x = params.minRealx + ix * params.dx;

            // Corresponds to the index of the cell to the left of footCoord
            const int LeftDiscreteNode =
                Kokkos::floor((xFootCoord - minRealx) * inv_dx);

            const double d_prev1 =
                LAG_OFFSET +
                inv_dx * (xFootCoord - coord(LeftDiscreteNode, minRealx, dx));

            auto coef = lag_basis(d_prev1);

            const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

            m_ftmp(i_fict, ivx, ix) = 0;   // initializing slice
            for (int k = 0; k <= LAG_ORDER; k++) {
                int idx_ipos1 = (nx + ipos1 + k) % nx;

                m_ftmp(i_fict, ivx, ix) +=
                    coef[k] * fdist(i_fict, ivx, idx_ipos1);
            }
        });

        Kokkos::deep_copy(m_ftmp, fdist);
}