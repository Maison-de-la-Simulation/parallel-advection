#include "advectors.h"

sycl::event
AdvX::Hierarchical::operator()(sycl::queue &Q, double *fdist_dev,
                               const Solver &solver) {
    const auto n0 = solver.p.n0;
    const auto n1 = solver.p.n1;
    const auto n2 = solver.p.n2;

    const sycl::range nb_wg{n0, 1, n2};
    const sycl::range wg_size{1, solver.p.pref_wg_size, 1};

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(n1), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{1, n1, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                    const int i1 = it.get_local_id(1);
                    const int i0 = g.get_group_id(0);
                    const int i2 = g.get_group_id(2);

                    auto slice = std::experimental::submdspan(
                        fdist, i0, std::experimental::full_extent, i2);

                    slice_ftmp[i1] = solver(slice, i0, i1, i2);
                });   // end parallel_for_work_item --> Implicit barrier
#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
            g.parallel_for_work_item(
                sycl::range{1, n1, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                    const int i1 = it.get_local_id(1);
                    const int i0 = g.get_group_id(0);
                    const int i2 = g.get_group_id(2);

                    fdist(i0, i1, i2) = slice_ftmp[i1];
                });
#else
            g.async_work_group_copy(
                sycl::multi_ptr<double,
                                sycl::access::address_space::global_space>(
                    fdist_dev) +
                    g.get_group_id(2) + g.get_group_id(0) * n2 * n1, /* dest */
                slice_ftmp.get_pointer(), /* source */
                n1,                       /* n elems */
                n2                        /* stride */
            );
#endif
        });   // end parallel_for_work_group
    });       // end Q.submit
}
