#include "advectors.h"

/* =================================================================
Same as Exp3 but buffer is allocated in local memory this time
- Problem, we allocate (n1*n2) in local memory which can be too much
==================================================================== */

// ==========================================
// ==========================================
sycl::event
AdvX::Exp4::operator()(sycl::queue &Q, double *fdist_dev,
                       const Solver &solver) {

    auto const n0 = solver.p.n0;
    auto const n1 = solver.p.n1;
    auto const n2 = solver.p.n2;

    /*=====================
      =====================
        In ctor
    =======================*/
    const sycl::range logical_wg(1, n1, n2);
    const sycl::range physical_wg(1, 1, 128);   // TODO: adapt for performance
    /*=====================
    =======================*/

    const sycl::range nb_wg(n0, 1, 1);

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 2> slice_ftmp(sycl::range<2>(n1, n2), cgh,
                                                   sycl::no_init);

        cgh.parallel_for_work_group(nb_wg, physical_wg, [=](sycl::group<3> g) {
            /* Solve kernel */
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                mdspan2d_t scr_view(slice_ftmp.get_pointer(), n1, n2);

                const int i1 = it.get_local_id(1);
                const int i2 = it.get_local_id(2);
                const int i0 = g.get_group_id(0);

                auto slice = std::experimental::submdspan(
                    fdist_view, i0, std::experimental::full_extent, i2);

                scr_view(i1, i2) = solver(i0, i1, i2, slice);

            });   // end parallel_for_work_item --> Implicit barrier

            /* Copy kernel*/
            // TODO: probably can use contiguous copy or something?
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                mdspan2d_t scr_view(slice_ftmp.get_pointer(), n1, n2);

                const int i1 = it.get_local_id(1);
                const int i2 = it.get_local_id(2);

                const int i0 = g.get_group_id(0);

                fdist_view(i0, i1, i2) = scr_view(i1, i2);
            });   // barrier
        });       // end parallel_for_work_group
    });           // end Q.submit
}