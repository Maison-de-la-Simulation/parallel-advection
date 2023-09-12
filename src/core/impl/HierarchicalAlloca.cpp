#include "advectors.h"

sycl::event
AdvX::HierarchicalAlloca::operator()(sycl::queue &Q,
                                     sycl::buffer<double, 2> &buff_fdistrib,
                                     const ADVParams &params) noexcept {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range<1> nb_wg{nvx};
    const sycl::range<1> wg_size{params.wg_size};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<1> g) {
            double *slice_ftmp = (double *) alloca(sizeof(double) * nx);

            g.parallel_for_work_item(
                sycl::range<1>(nx), [&](sycl::h_item<1> it) {
                    const int ix = it.get_local_id(0);
                    const int ivx = g.get_group_id(0);

                    double const xFootCoord = displ(ix, ivx, params);

                    const int LeftDiscreteNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(LeftDiscreteNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

                    slice_ftmp[ix] = 0;   // initializing slice for each work
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            g.async_work_group_copy(fdist.get_pointer() +
                                        nx * g.get_group_id(0),
                                    sycl::local_ptr<double>(slice_ftmp), nx)
                .wait();
            // code executed only once
        });   // end parallel_for_work_group
    });       // end Q.submit
}