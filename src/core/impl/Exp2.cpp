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
AdvX::Exp2::actual_advection(sycl::queue &Q, buff3d &buff_fdistrib,
                             const ADVParams &params,
                             const size_t &ny_batch_size,
                             const size_t &ny_offset) {

    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    auto const wg_size_y = params.wg_size_y;
    auto const wg_size_x = params.wg_size_x;

    /* ny must be divisible by slice_size_dim_y */
    if (ny_batch_size % wg_size_y != 0) {
        throw std::invalid_argument(
            "ny_batch_size must be divisible by wg_size_y");
    }
    if (wg_size_y * nx > 6144) {
        throw std::invalid_argument(
            "wg_size_y*nx must be < to 6144 (shared memory limit)");
    }

    const sycl::range nb_wg{ny_batch_size / wg_size_y, 1, ny1};
    const sycl::range wg_size{wg_size_y, wg_size_x, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_y, nx), cgh, sycl::no_init);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            /* Copy kernel*/
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                    mdspan2d_t localAcc_view(slice_ftmp.get_pointer(),
                                             slice_ftmp.get_range().get(0),
                                             slice_ftmp.get_range().get(1));

                    const int ix = it.get_local_id(1);
                    const int iy1 = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int iy =
                        wg_size_y * g.get_group_id(0) + ny_offset + local_ny;

                    localAcc_view(local_ny, ix) = fdist_view(iy, ix, iy1);
                });   // barrier

            /* Solve kernel */
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                    mdspan2d_t localAcc_view(slice_ftmp.get_pointer(),
                                             slice_ftmp.get_range().get(0),
                                             slice_ftmp.get_range().get(1));

                    const int ix = it.get_local_id(1);
                    const int iy1 = g.get_group_id(2);

                    const int local_ny = it.get_local_id(0);
                    const int iy =
                        wg_size_y * g.get_group_id(0) + ny_offset + local_ny;

                    double const xFootCoord = displ(ix, iy, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    fdist_view(iy, ix, iy1) = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        fdist_view(iy, ix, iy1) +=
                            coef[k] * localAcc_view(local_ny, idx_ipos1);
                    }
                });   // end parallel_for_work_item --> Implicit barrier
        });           // end parallel_for_work_group
    });               // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp2::operator()(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                       const ADVParams &params) {

    // can be parallel on multiple streams?
    for (size_t i_batch = 0; i_batch < n_batch_ - 1; ++i_batch) {

        size_t ny_offset = (i_batch * MAX_NY_BATCHS);

        actual_advection(Q, buff_fdistrib, params, MAX_NY_BATCHS, ny_offset)
            .wait();
    }

    // return the last advection with the rest
    return actual_advection(Q, buff_fdistrib, params, last_ny_size_,
                            last_ny_offset_);
    // }
}