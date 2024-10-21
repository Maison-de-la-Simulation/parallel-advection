#include "IAdvectorX.h"
#include "advectors.h"

/* =================================================================
Same as Exp3 but buffer is allocated in local memory this time
- Problem, we allocate (n1*n2) in local memory which can be too much
==================================================================== */

// ==========================================
// ==========================================
sycl::event
AdvX::Exp4::operator()(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                       const ADVParams &params) {

    auto const n1 = params.n1;
    auto const n0 = params.n0;
    auto const n2 = params.n2;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

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
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(n1, n2), cgh, sycl::no_init);

        cgh.parallel_for_work_group(nb_wg, physical_wg, [=](sycl::group<3> g) {
            /* Solve kernel */
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist.get_pointer(), n0, n1, n2);
                mdspan2d_t scr_view(slice_ftmp.get_pointer(), n1, n2);

                const int i1 = it.get_local_id(1);
                const int i2 = it.get_local_id(2);

                const int i0 = g.get_group_id(0);

                double const xFootCoord = displ(i1, i0, params);

                // index of the cell to the left of footCoord
                const int leftNode =
                    sycl::floor((xFootCoord - minRealX) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET +
                    inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                auto coef = lag_basis(d_prev1);

                const int ipos1 = leftNode - LAG_OFFSET;

                scr_view(i1, i2) = 0.;
                for (int k = 0; k <= LAG_ORDER; k++) {
                    int id1_ipos = (n1 + ipos1 + k) % n1;

                    scr_view(i1, i2) +=
                        coef[k] * fdist_view(i0, id1_ipos, i2);
                }
            });   // end parallel_for_work_item --> Implicit barrier

            /* Copy kernel*/
            // TODO: probably can use contiguous copy or something?
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist.get_pointer(), n0, n1, n2);
                mdspan2d_t scr_view(slice_ftmp.get_pointer(), n1, n2);

                const int i1 = it.get_local_id(1);
                const int i2 = it.get_local_id(2);

                const int i0 = g.get_group_id(0);

                fdist_view(i0, i1, i2) = scr_view(i1, i2);
            });   // barrier
        });       // end parallel_for_work_group
    });           // end Q.submit
}