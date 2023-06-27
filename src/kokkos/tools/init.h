#pragma once

#include "unique_ref.h"
#include <AdvectionParams.h>
#include <InitParams.h>
#include <x_advectors.h>
#include "../kokkos_shortcut.hpp"
// #include <vx_advectors.h>

// To switch case on a str
[[nodiscard]] constexpr unsigned int
str2int(const char *str, int h = 0) noexcept {
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

static constexpr auto error_str =
    "Should be one of: {MDRange}";

// // ==========================================
// // ==========================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
x_advector_factory(const ADVParams &adParams, const InitParams &initParams) {
    std::string kernel_name = initParams.kernelImpl.data();

    switch (str2int(kernel_name.data())) {
    case str2int("MDRange"):
        return sref::make_unique<advector::x::MDRange>();
    default:
        auto str = kernel_name + " is not a valid kernel name.\n" + error_str;
        throw std::runtime_error(str);
    }
}

// // ==========================================
// // ==========================================
// [[nodiscard]] sref::unique_ref<IAdvectorVx>
// vx_advector_factory() {
// //    return sref::make_unique<advector::vx::Hierarchical>();
// }

// ==========================================
// ==========================================
void
fill_buffers(KV_double_3d &fdist, KV_double_1d &efield,
             const ADVParams &params) noexcept {

        Kokkos::Array<int, 3> begin{0,0,0};
        Kokkos::Array<int, 3> end{fdist.extent_int(0),
                                  fdist.extent_int(1),
                                  fdist.extent_int(2)};
        // Kokkos::Array<int, 3> end{fdist.extent(0),
        //                           fdist.extent(1),
        //                           fdist.extent(2)};

        Kokkos::parallel_for(
        "fill_buffers",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            const int ix = k;
            const double x = params.minRealx + ix * params.dx;

            fdist(i, j, k) = Kokkos::sin(4 * x * Kokkos::numbers::pi);

            efield(ix) = 0;
        });
}