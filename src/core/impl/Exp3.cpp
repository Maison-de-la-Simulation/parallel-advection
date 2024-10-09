#include "IAdvectorX.h"
#include "advectors.h"

/* =================================================================
Scratch is fully in global memory, parallelizing on Y1 rather than Y

- Each WI will be placed on the Y1 dimension, iterating sequentially through X
- Parameter p, controlling the number of Y slices done concurrently
- Buffer of size concurrent_ny_slices_*nx*ny1 is allocated
    - Controlling memory footprint/perf with p

==================================================================== */

// ==========================================
// ==========================================
sycl::event
AdvX::Exp3::actual_advection(sycl::queue &Q, buff3d &buff_fdistrib,
                             const ADVParams &params,
                             const size_t &ny_batch_size,
                             const size_t &ny_offset) {

    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    /*=====================
      =====================
        In ctor
    =======================*/
    // TODO: careful check size is not excedding max work group size
    const sycl::range logical_wg(1, nx, ny1);
    const sycl::range physical_wg(1, 1, 128);   // TODO: adapt for performance

    /*=====================
    =======================*/
    const sycl::range nb_wg(ny_batch_size, 1, 1);

    const auto scratch = scratch_;
    const auto concurrent_ny_slice = concurrent_ny_slices_;

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        /*TODO: use the local accessor if it's possible (i.e. if nx*ny1 <
         * MAX_ALLOC) that's Exp4 */
        // sycl::local_accessor<double, 2> slice_ftmp(sycl::range<2>(wg_size_y,
        // nx), cgh, sycl::no_init);

        cgh.parallel_for_work_group(nb_wg, physical_wg, [=](sycl::group<3> g) {
            /* Solve kernel */
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                mdspan3d_t scr_view(scratch, concurrent_ny_slice, nx, ny1);

                const int ix = it.get_local_id(1);
                const int iy1 = it.get_local_id(2);

                const int scr_iy = g.get_group_id(0);
                const int iy = scr_iy + ny_offset;

                double const xFootCoord = displ(ix, iy, params);

                // index of the cell to the left of footCoord
                const int leftNode =
                    sycl::floor((xFootCoord - minRealX) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET +
                    inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                auto coef = lag_basis(d_prev1);

                const int ipos1 = leftNode - LAG_OFFSET;

                scr_view(scr_iy, ix, iy1) = 0.;
                for (int k = 0; k <= LAG_ORDER; k++) {
                    int idx_ipos1 = (nx + ipos1 + k) % nx;

                    scr_view(scr_iy, ix, iy1) +=
                        coef[k] * fdist_view(iy, idx_ipos1, iy1);
                }
            });   // end parallel_for_work_item --> Implicit barrier

            /* Copy kernel*/
            // TODO: probably can use contiguous copy or something?
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                mdspan3d_t scr_view(scratch, concurrent_ny_slice, nx, ny1);

                const int ix = it.get_local_id(1);
                const int iy1 = it.get_local_id(2);

                const int scr_iy = g.get_group_id(0);
                const int iy = scr_iy + ny_offset;

                fdist_view(iy, ix, iy1) = scr_view(scr_iy, ix, iy1);
            });   // barrier
        });       // end parallel_for_work_group
    });           // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp3::operator()(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                       const ADVParams &params) {

    // can be parallel on multiple streams?
    for (size_t i_batch = 0; i_batch < n_batch_ - 1; ++i_batch) {

        size_t ny_offset = (i_batch * concurrent_ny_slices_);

        actual_advection(Q, buff_fdistrib, params, concurrent_ny_slices_,
                         ny_offset)
            .wait();
        /* on est obligé de wait à moins d'utiliser plein de petits buffers et
        de soumettre plein de tout petit noyaux
        comme des local_accessor et dire quand la petite slice est process pour
        liberer la petite slice en memoire */
    }

    // return the last advection with the rest
    return actual_advection(Q, buff_fdistrib, params, last_ny_size_,
                            last_ny_offset_);
    // }
}