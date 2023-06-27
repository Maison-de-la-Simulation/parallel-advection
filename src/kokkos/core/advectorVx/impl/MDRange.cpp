#include "vx_advectors.h"

void
advector::vx::MDRange::operator()(KV_double_3d &fdist, KV_double_1d &elec_field,
                                  const ADVParams &params) noexcept {

    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const n_fict = params.n_fict_dim;
    auto const minRealVx = params.minRealVx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const inv_dvx = params.inv_dvx;
    auto const realWidthVx = params.realWidthVx;

    // const sycl::range<3> nb_wg{n_fict, 1, nx};
    // const sycl::range<3> wg_size{1, params.wg_size, 1};

    // return Q.submit([&](sycl::handler &cgh) {
    //     auto fdist =
    //         buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);
    //     auto efield =
    //         buff_electric_field.get_access<sycl::access::mode::read>(cgh);

    //     sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>{nvx}, cgh);

    //     cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
    //         const int ix = g.get_group_id(2);
    //         const int i_fict = g.get_group_id(0);

    //         g.parallel_for_work_item(
    //             sycl::range<3>(1, nvx, 1), [&](sycl::h_item<3> it) {
    //                 const int ivx = it.get_local_id(1);

    //                 auto const ex = efield[ix];
    //                 auto const displx = dt * ex;
    //                 double const vx = coord(ivx, minRealVx, dvx);

    //                 auto const vxFootCoord =
    //                     minRealVx +
    //                     sycl::fmod(realWidthVx + vx - displx - minRealVx,
    //                                realWidthVx);

    //                 const int LeftDiscreteNode =
    //                     sycl::floor((vxFootCoord - minRealVx) * inv_dvx);

    //                 const double d_prev1 =
    //                     LAG_OFFSET +
    //                     inv_dvx * (vxFootCoord -
    //                               coord(LeftDiscreteNode, minRealVx, dvx));

    //                 auto coef = lag_basis(d_prev1);

    //                 const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

    //                 slice_ftmp[ivx] = 0;
    //                 for (int k = 0; k <= LAG_ORDER; k++) {
    //                     int idx_coef = (nvx + ipos1 + k) % nvx;

    //                     slice_ftmp[ivx] += coef[k] *
    //                     fdist[i_fict][idx_coef][ix];
    //                 }
    //             });   // end parallel_for_work_item --> Implicit barrier

    //         //Copy not contiguous slice
    //         g.parallel_for_work_item(sycl::range<3>(1, nvx, 1),
    //                                  [&](sycl::h_item<3> it) {
    //                                      const int ivx = it.get_local_id(1);

    //                                      fdist[i_fict][ivx][ix] =
    //                                      slice_ftmp[ivx];
    //                                  });

    //     });   // end parallel_for_work_group
    // });       // end Q.submit
}