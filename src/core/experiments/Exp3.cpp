#include "advectors.h"

/* =================================================================
Scratch is fully in global memory, parallelizing on Y1 rather than Y

- Each WI will be placed on the Y1 dimension, iterating sequentially through X
- Parameter p, controlling the number of Y slices done concurrently
- Buffer of size concurrent_ny_slices_*n1*n2 is allocated
    - Controlling memory footprint/perf with p

==================================================================== */

// ==========================================
// ==========================================
sycl::event
AdvX::Exp3::actual_advection(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver, const size_t &ny_batch_size,
                             const size_t &ny_offset) {

    auto const n0 = solver.p.n0;
    auto const n1 = solver.p.n1;
    auto const n2 = solver.p.n2;

    /*=====================
      =====================
        In ctor
    =======================*/
    // TODO: careful check size is not excedding max work group size
    const sycl::range logical_wg(1, n1, n2);
    const sycl::range physical_wg(1, 1, 128);   // TODO: adapt for performance

    /*=====================
    =======================*/
    const sycl::range nb_wg(ny_batch_size, 1, 1);

    const auto scratch = scratch_;
    const auto concurrent_ny_slice = concurrent_ny_slices_;

    return Q.submit([&](sycl::handler &cgh) {
        /*TODO: use the local accessor if it's possible (i.e. if n1*n2 <
         * MAX_ALLOC) that's Exp4 */
        // sycl::local_accessor<double, 2> slice_ftmp(sycl::range<2>(wg_size_0,
        // n1), cgh, sycl::no_init);

        cgh.parallel_for_work_group(nb_wg, physical_wg, [=](sycl::group<3> g) {
            /* Solve kernel */
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                mdspan3d_t scr_view(scratch, concurrent_ny_slice, n1, n2);

                const int i1 = it.get_local_id(1);
                const int i2 = it.get_local_id(2);

                const int scr_iy = g.get_group_id(0);
                const int i0 = scr_iy + ny_offset;

                auto slice = std::experimental::submdspan(
                    fdist_view, i0, std::experimental::full_extent, i2);

                scr_view(scr_iy, i1, i2) = solver(slice, i0, i1, i2);
            });   // end parallel_for_work_item --> Implicit barrier

            /* Copy kernel*/
            // TODO: probably can use contiguous copy or something?
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                mdspan3d_t scr_view(scratch, concurrent_ny_slice, n1, n2);

                const int i1 = it.get_local_id(1);
                const int i2 = it.get_local_id(2);

                const int scr_iy = g.get_group_id(0);
                const int i0 = scr_iy + ny_offset;

                fdist_view(i0, i1, i2) = scr_view(scr_iy, i1, i2);
            });   // barrier
        });       // end parallel_for_work_group
    });           // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp3::operator()(sycl::queue &Q, double *fdist_dev,
                       const Solver &solver) {

    // can be parallel on multiple streams?
    for (size_t i_batch = 0; i_batch < n_batch_ - 1; ++i_batch) {

        size_t ny_offset = (i_batch * concurrent_ny_slices_);

        actual_advection(Q, fdist_dev, solver, concurrent_ny_slices_, ny_offset)
            .wait();
        /* on est obligé de wait à moins d'utiliser plein de petits buffers et
        de soumettre plein de tout petit noyaux
        comme des local_accessor et dire quand la petite slice est process pour
        liberer la petite slice en memoire */
    }

    // return the last advection with the rest
    return actual_advection(Q, fdist_dev, solver, last_n0_size_,
                            last_n0_offset_);
    // }
}