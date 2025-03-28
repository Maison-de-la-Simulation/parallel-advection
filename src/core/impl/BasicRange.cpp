#include "advectors.hpp"

sycl::event
AdvX::BasicRange::operator()(sycl::queue &Q, real_t *fdist_dev,
                             const AdvectionSolver &solver) {
    auto const n0 = solver.params.n0;
    auto const n1 = solver.params.n1;
    auto const n2 = solver.params.n2;

    sycl::range r3d(n0, n1, n2);

    Q.submit([&](sycl::handler &cgh) {
        /* Using the preallocated global buffer */
        span3d_t ftmp(ftmp_, n0, n1, n2);

        cgh.parallel_for(r3d, [=](sycl::id<3> itm) {
            span3d_t fdist(fdist_dev, n0, n1, n2);
            const int i1 = itm[1];
            const int i0 = itm[0];
            const int i2 = itm[2];

            ftmp(i0, i1, i2) =
                solver(std::experimental::submdspan(
                           fdist, i0, std::experimental::full_extent, i2),
                       i0, i1, i2);
            // barrier
        });   // end parallel_for
    });       // end Q.submit
    Q.wait();
    //copy
    return Q.submit([&](sycl::handler &cgh) {
        span3d_t ftmp(ftmp_, n0, n1, n2);
        cgh.parallel_for(r3d, [=](sycl::id<3> itm) {
            span3d_t fdist(fdist_dev, n0, n1, n2);
            const int i1 = itm[1];
            const int i0 = itm[0];
            const int i2 = itm[2];
            fdist(i0, i1, i2) = ftmp(i0, i1, i2);
            // barrier
        });   // end parallel_for
    });   // end Q.submit
}
