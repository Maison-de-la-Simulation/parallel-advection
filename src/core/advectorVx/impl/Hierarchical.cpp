#include "vx_advectors.h"

sycl::event
advector::vx::Hierarchical::operator()(
    sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
    sycl::buffer<double, 1> &buff_electric_field,
    const ADVParams &params) noexcept {

    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const minRealVx = params.minRealVx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const inv_dvx = params.inv_dvx;
    auto const realWidthVx = params.realWidthVx;

    const sycl::range<1> nb_wg{nx};
    const sycl::range<1> wg_size{params.wg_size};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);
        auto efield =
            buff_electric_field.get_access<sycl::access::mode::read>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>{nvx}, cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<1> g) {
            g.parallel_for_work_item(
                sycl::range<1>(nvx), [&](sycl::h_item<1> it) {
                    const int ix  = g.get_group_id(0);
                    const int ivx = it.get_local_id(0);

                    auto const ex = efield[ix];
                    auto const displx = dt * ex;
                    double const vx = coord(ivx, minRealVx, dvx);

                    auto const vxFootCoord =
                        minRealVx +
                        sycl::fmod(realWidthVx + vx - displx - minRealVx,
                                   realWidthVx);

                    const int LeftDiscreteNode =
                        sycl::floor((vxFootCoord - minRealVx) * inv_dvx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dvx * (vxFootCoord -
                                  coord(LeftDiscreteNode, minRealVx, dvx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

                    slice_ftmp[ivx] = 0;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_coef = (nvx + ipos1 + k) % nvx;

                        slice_ftmp[ivx] += coef[k] * fdist[idx_coef][ix];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            //Copy not contiguous slice
            g.parallel_for_work_item(sycl::range<1>(nvx),
                                     [&](sycl::h_item<1> it) {
                                         const int ix = g.get_group_id(0);
                                         const int ivx = it.get_local_id(0);

                                         fdist[ivx][ix] = slice_ftmp[ivx];
                                     });

        });   // end parallel_for_work_group
    });       // end Q.submit
}