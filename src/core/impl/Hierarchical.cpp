#include "advectors.h"

sycl::event
AdvX::Hierarchical::operator()(sycl::queue &Q,
                               sycl::buffer<double, 2> &buff_fdistrib,
                               const ADVParams &params) const {
    auto const nx = params.nx;
    auto const nVx = params.nVx;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    assert(nVx % 512 == 0);
    const sycl::range<2> nb_wg{nx, nVx / 512};
    const sycl::range<2> wg_size{1, 512};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>{512}, cgh);
        // double slice_ftmp[512];
        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<2> g) {
            g.parallel_for_work_item([&](sycl::h_item<2> it) {
                const int ix = g.get_group_id(0);
                const int ivx = it.get_local_id(1);

                double const xFootCoord = displ(ix, ivx, params);

                // Corresponds to the index of the cell to the left of
                // footCoord
                const int leftDiscreteCell =
                    sycl::floor((xFootCoord - minRealx) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET +
                    inv_dx * (xFootCoord - (minRealx + leftDiscreteCell * dx));

                double coef[LAG_PTS];
                lag_basis(d_prev1, coef);

                const int ipos1 = leftDiscreteCell - LAG_OFFSET;

                double ftmp = 0.;
                // slice_ftmp[ix] = 0;   // initializing slice for each work item
                for (int k = 0; k <= LAG_ORDER; k++) {
                    int idx_ipos1 = (nx + ipos1 + k) % nx;

                    // slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1];
                    ftmp += coef[k] * fdist[idx_ipos1][ivx];
                }

                slice_ftmp[ivx] = ftmp;
            });   // end parallel_for_work_item --> Implicit barrier

            // fdist[g.get_group_id(0)][] = ftmp;
            g.parallel_for_work_item([&](sycl::h_item<2> it) {
                fdist[g.get_group_id(0)][it.get_local_id(1)] = slice_ftmp[it.get_local_id(1)];
            });

            //code executed only once
        });   // end parallel_for_work_group
    });       // end Q.submit
}