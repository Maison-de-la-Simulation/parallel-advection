#include "advectors.h"

sycl::event
AdvX::HierarchicalAlloca::operator()(sycl::queue &Q,
                                     sycl::buffer<double, 2> &buff_fdistrib,
                                     const ADVParams &params) const noexcept {
    auto const nx = params.nx;
    auto const nVx = params.nVx;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range<2> nb_wg{nVx, 1};
    const sycl::range<2> wg_size{1, 512};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<2> g) {
            // Should be executed only once
            auto slice_ftmp = (double *) alloca(sizeof(double) * nx);

            g.parallel_for_work_item(
                sycl::range<2>(1, nx), [&](sycl::h_item<2> it) {
                    const int ix = it.get_global_id(1);
                    const int ivx = g.get_group_id(0);

                    double const xFootCoord = displ(ix, ivx, params);

                    // Corresponds to the index of the cell to the left of
                    // footCoord
                    const int leftDiscreteCell =
                        sycl::floor((xFootCoord - minRealx) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(leftDiscreteCell, minRealx, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftDiscreteCell - LAG_OFFSET;

                    slice_ftmp[ix] = 0;   // initializing slice for each work
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            // fdist[g.get_group_id(0)][] = ftmp;
            g.parallel_for_work_item(sycl::range<2>(1, nx),
                                     [&](sycl::h_item<2> it) {
                                         const int ix = it.get_global_id(1);
                                         const int ivx = it.get_global_id(0);
                                         fdist[ivx][ix] = slice_ftmp[ix];
                                         // fdist[g.get_group_id(0)][g.get_group_id(1)]
                                         // = slice_ftmp[it.get_local_id(1)];
                                     });

            // g.async_work_group_copy();

            // code executed only once
        });   // end parallel_for_work_group
    });       // end Q.submit
}