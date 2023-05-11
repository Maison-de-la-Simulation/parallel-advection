#include "advectors.h"

sycl::event
AdvX::Scoped::operator()(
    sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib) const noexcept {
    auto const nx = m_params.nx;
    auto const nVx = m_params.nVx;
    auto const minRealx = m_params.minRealx;
    auto const dx = m_params.dx;
    auto const inv_dx = m_params.inv_dx;

    sycl::range<2> nb_wg{nVx, 1};
    sycl::range<2> wg_size{1, nx};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);

        // slice_ftmp.get_pointer()

        // cgh.copy()
        cgh.parallel(nb_wg, wg_size, [=](auto g) {
            // c.f.
            // https://github.com/OpenSYCL/OpenSYCL/blob/develop/doc/scoped-parallelism.md#memory-placement-rules
            //   double slice_ftmp[nx];   // declared in the private memory of
            //   the executing physical WI ???
            // Actually doesn't work if version of CUDA is not 11.6. I have to
            // use the local_accessor

            const int ivx = g.get_group_id(0);

            sycl::device_event e = g.async_work_group_copy(
                slice_ftmp.get_pointer(), fdist.get_pointer() + nx * ivx, nx);

            e.wait();   // let's be sure the slice is nicely copied

            sycl::distribute_groups_and_wait(g, [&](auto subg) {
                sycl::distribute_items_and_wait(subg, [&](sycl::s_item<2> it) {
                    const int ix = it.get_local_id(g, 1);
                    const int ivx = g.get_group_id(0);
                    // const int ivx = g.get_group_id(1) * 32 +
                    // it.get_local_id(g,1);

                    double const xFootCoord = displ(ix, ivx, m_params);

                    // Corresponds to the index of the cell to
                    // the left of footCoord
                    const int LeftDiscreteNode =
                        sycl::floor((xFootCoord - minRealx) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(LeftDiscreteNode, minRealx, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = LeftDiscreteNode - LAG_OFFSET;
                    double ftmp = 0.;

                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;
                        ftmp += coef[k] * slice_ftmp[idx_ipos1];
                    }

                    fdist[ivx][ix] = ftmp;
                });   // end distribute items
            });       // end distribute_groups
        });           // end parallel regions
    });               // end Q.submit
}