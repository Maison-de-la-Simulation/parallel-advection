#include "advectors.h"

sycl::event
AdvX::BasicRange::operator()(sycl::queue &Q, double *fdist_dev,
                             const AdvectionSolver &solver) {
    auto const n0 = solver.params.n0;
    auto const n1 = solver.params.n1;
    auto const n2 = solver.params.n2;

    sycl::range r3d(n0, n1, n2);

    Q.submit([&](sycl::handler &cgh) {
        /* Using the preallocated global buffer */
        sycl::accessor ftmp(m_global_buff_ftmp, cgh, sycl::write_only,
                            sycl::no_init);

        cgh.parallel_for(r3d, [=](sycl::id<3> itm) {
            mdspan3d_t fdist(fdist_dev, r3d.get(0), r3d.get(1), r3d.get(2));
            const int i1 = itm[1];
            const int i0 = itm[0];
            const int i2 = itm[2];

            ftmp[i0][i1][i2] =
                solver(std::experimental::submdspan(
                           fdist, i0, std::experimental::full_extent, i2),
                       i0, i1, i2);
            // barrier
        });   // end parallel_for
    });       // end Q.submit

    return Q.submit([&](sycl::handler &cgh) {
        auto ftmp =
            m_global_buff_ftmp.get_access<sycl::access::mode::read>(cgh);
        cgh.copy(ftmp, fdist_dev);
    });   // end Q.submit
}
