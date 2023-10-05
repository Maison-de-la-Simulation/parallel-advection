#include "advectors.h"

sycl::event
AdvX::Hierarchical::operator()(sycl::queue &Q,
                               sycl::buffer<double, 2> &buff_fdistrib,
                               const ADVParams &params) noexcept {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    // auto const minRealX = params.minRealX;
    auto const minRealVx = params.minRealVx;
    // auto const dx = params.dx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const realWidthX = params.realWidthX;

    // assert(nvx % 512 == 0);
    const sycl::range<1> nb_wg{nvx};
    const sycl::range<1> wg_size{params.wg_size};


    return Q.submit([&](sycl::handler &cgh) {
#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
throw std::logic_error("Hierarchical kernel is not compatible with DPCPP");
#else   // for acpp
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);
        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<1> g) {
            g.parallel_for_work_item(
                sycl::range<1>(nx), [&](sycl::h_item<1> it) {
                    const int ix = it.get_local_id(0);
                    const int ivx = g.get_group_id(0);
                    // double const xFootCoord = displ(ix, ivx, params);
                    double const xFootCoord = displ(
                        ix, ivx, minRealX, minRealVx, realWidthX, dx, dvx, dt);

                    // Corresponds to the index of the cell to the left of footCoord
                    const int LeftDiscreteNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(LeftDiscreteNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

                    // double ftmp = 0.;
                    slice_ftmp[ix] = 0;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            g.async_work_group_copy(fdist.get_pointer() +
                                        nx * g.get_group_id(0),
                                    slice_ftmp.get_pointer(), nx)
                .wait();

        });   // end parallel_for_work_group
#endif
    });       // end Q.submit
}