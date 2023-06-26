#include "x_advectors.h"

sycl::event
advector::x::Scoped::operator()(sycl::queue &Q,
                                sycl::buffer<double, 3> &buff_fdistrib,
                                const ADVParams &params) noexcept {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const n_fict = params.n_fict_dim;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range<3> nb_wg{n_fict, nvx, 1};
    const sycl::range<3> wg_size{1, 1, nx};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);

        cgh.parallel(nb_wg, wg_size, [=](auto g) {
            const int ivx = g.get_group_id(1);
            const int i_fict = g.get_group_id(0);

            sycl::distribute_items_and_wait(g, [&](sycl::s_item<3> it) {
                const int ix = it.get_local_id(g, 2);

                double const xFootCoord = displ(ix, ivx, params);

                // Corresponds to the index of the cell to
                // the left of footCoord
                const int LeftDiscreteNode =
                    sycl::floor((xFootCoord - minRealx) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET + inv_dx * (xFootCoord - coord(LeftDiscreteNode,
                                                              minRealx, dx));

                auto coef = lag_basis(d_prev1);

                const int ipos1 = LeftDiscreteNode - LAG_OFFSET;
                // double ftmp = 0.;
                slice_ftmp[ix] = 0.;
                for (int k = 0; k <= LAG_ORDER; k++) {
                    int idx_ipos1 = (nx + ipos1 + k) % nx;
                    // ftmp += coef[k] * slice_ftmp[idx_ipos1];
                    slice_ftmp[ix] += coef[k] * fdist[i_fict][ivx][idx_ipos1];
                }

                // fdist[ivx][ix] = ftmp;
            });   // end distribute items
                  // });       // end distribute_groups

            g.async_work_group_copy(fdist.get_pointer() +
                                        nx * ivx + nx*nvx*i_fict,
                                    slice_ftmp.get_pointer(), nx)
                .wait();
        });   // end parallel regions
    });       // end Q.submit
}