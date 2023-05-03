#include "advectors.h"

sycl::event
AdvX::Hierarchical::operator()(sycl::queue &Q,
                               sycl::buffer<double, 2> &buff_fdistrib,
                               const ADVParams &params) const noexcept {
    auto const nx = params.nx;
    auto const nVx = params.nVx;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    // assert(nVx % 512 == 0);
    const sycl::range<2> nb_wg{nVx, 1};
    const sycl::range<2> wg_size{1, 1024};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>{nx}, cgh);
        // double slice_ftmp[512];vi
        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<2> g) {
            g.parallel_for_work_item(
                sycl::range<2>(1, nx), [&](sycl::h_item<2> it) {
                    const int ix = it.get_global_id(1);
                    //g.get_group_id(0) also works for ivx
                    const int ivx = it.get_global_id(0);

                    double const xFootCoord = displ(ix, ivx, params);

                    // Corresponds to the index of the cell to the left of
                    // footCoord
                    const int LeftDiscreteNode =
                        sycl::floor((xFootCoord - minRealx) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(LeftDiscreteNode, minRealx, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

                    // double ftmp = 0.;
                    // fdist[ivx][ix] = 0;
                    slice_ftmp[ix] = 0;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        // fdist[ivx][ix] += coef[k] * slice_ftmp[idx_ipos1];
                        slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            const int ivx = g.get_group_id(0);
            g.async_work_group_copy(fdist.get_pointer() + ivx * nx,
                                    slice_ftmp.get_pointer(), nx).wait();
                                    
        });   // end parallel_for_work_group
    });       // end Q.submit
}