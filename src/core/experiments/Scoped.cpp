#include "advectors.h"

sycl::event
AdvX::Scoped::operator()(sycl::queue &Q,double* fdist_dev,
                         const Solver &solver) {
    auto const n0 = solver.p.n0;
    auto const n1 = solver.p.n1;
    auto const n2 = solver.p.n2;

    const sycl::range nb_wg{n0, 1, n2};
    const sycl::range wg_size{1, n1, 1};

    return Q.submit([&](sycl::handler &cgh) {
#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
throw std::logic_error("Scoped kernel is not compatible with DPCPP");
#else   // for acpp
        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(n1), cgh);
         
        cgh.parallel(nb_wg, wg_size, [=](auto g) {
                sycl::distribute_items_and_wait(g, [&](auto /*sycl::s_item<3>*/ it) {
                    mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                    const int i0 = g.get_group_id(0);
                    const int i1 = it.get_local_id(g, 1);
                    const int i2 = g.get_group_id(2);
                    
                    auto slice = std::experimental::submdspan(
                        fdist, i0, std::experimental::full_extent, i2);

                    slice_ftmp[i1] = solver(slice, i0, i1, i2);
                });   // end distribute items

                g.async_work_group_copy(
                    sycl::multi_ptr<double,
                                    sycl::access::address_space::global_space>(
                        fdist_dev) +
                        g.get_group_id(2) +
                        g.get_group_id(0) * n2 * n1,   // dest
                    slice_ftmp.get_pointer(),          // source
                    n1,                                /* n elems */
                    n2                                 /* stride */
                );
        }); // end parallel regions
#endif
    });// end Q.submit
}
