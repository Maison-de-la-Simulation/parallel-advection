#include "x_advectors.h"

void
advector::x::MDRange::operator()(KV_double_3d &fdistrib,
                                 const ADVParams &params) noexcept {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    Kokkos::parallel_for(
    "MDrange_advection",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
    KOKKOS_CLASS_LAMBDA(int i, int j, int k)
    {
        const int ix = k;
        const double x = params.minRealx + ix * params.dx;

        fdist(i, j, k) = Kokkos::sin(4 * x * Kokkos::numbers::pi);

        efield(ix) = 0;
    });

}