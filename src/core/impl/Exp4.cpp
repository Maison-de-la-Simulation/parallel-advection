#include "IAdvectorX.h"
#include "advectors.h"
#include <experimental/mdspan>

using mdspan3d_t =
    std::experimental::mdspan<double, std::experimental::dextents<size_t, 3>,
                              std::experimental::layout_right>;
using mdspan2d_t =
    std::experimental::mdspan<double, std::experimental::dextents<size_t, 2>,
                              std::experimental::layout_right>;

// ==========================================
// ==========================================
sycl::event
AdvX::Exp4::operator()(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                       const ADVParams &params) {

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
    const sycl::range nb_wg(ny, 1, 1);

    // const auto scratch = scratch_;
    // const auto concurrent_ny_slice = concurrent_ny_slices_;

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(nx, ny1), cgh, sycl::no_init);

        cgh.parallel_for_work_group(nb_wg, physical_wg, [=](sycl::group<3> g) {
            /* Solve kernel */
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                mdspan2d_t scr_view(slice_ftmp.get_pointer(), nx, ny1);

                const int ix = it.get_local_id(1);
                const int iy1 = it.get_local_id(2);

                const int iy = g.get_group_id(0);

                double const xFootCoord = displ(ix, iy, params);

                // index of the cell to the left of footCoord
                const int leftNode =
                    sycl::floor((xFootCoord - minRealX) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET +
                    inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                auto coef = lag_basis(d_prev1);

                const int ipos1 = leftNode - LAG_OFFSET;

                scr_view(ix, iy1) = 0.;
                for (int k = 0; k <= LAG_ORDER; k++) {
                    int idx_ipos1 = (nx + ipos1 + k) % nx;

                    scr_view(ix, iy1) +=
                        coef[k] * fdist_view(iy, idx_ipos1, iy1);
                }
            });   // end parallel_for_work_item --> Implicit barrier

            /* Copy kernel*/
            // TODO: probably can use contiguous copy or something?
            g.parallel_for_work_item(logical_wg, [&](sycl::h_item<3> it) {
                mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                mdspan2d_t scr_view(slice_ftmp.get_pointer(), nx, ny1);

                const int ix = it.get_local_id(1);
                const int iy1 = it.get_local_id(2);

                const int iy = g.get_group_id(0);

                fdist_view(iy, ix, iy1) = scr_view(ix, iy1);
            });   // barrier
        });       // end parallel_for_work_group
    });           // end Q.submit
}