#include "IAdvectorX.h"
#include "advectors.h"

/* =================================================================
Vertical distribution of data between local and global memory
- Two types of kernels
    - Ones working in global memory
    - Ones working in local memory
- No modulo in accessor, each kernels works in different buffer
- Parameter p (percent_in_local_mem) what is the percentage of kernels to run
with local memory, the rest will be allocated in global memory
    - Limitation: for now p is only calculated wrt Y size, should be Y*Y1?
    - Example: Y = 100, p=0.6
        - 60 kernels will run with local memory
        - 40 kernels will run into global memory (3D buffer of size 40*nx*ny1
            is allocated)



- Streaming in Y with blocks BY
- GridStride (Done by hierarchical) in BY and X dims
==================================================================== */

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

    const sycl::range nb_wg_local {k_local_  / wg_size_y, 1, ny1};
    const sycl::range nb_wg_global{k_global_ / wg_size_y, 1, ny1};

    const sycl::range wg_size{wg_size_y, wg_size_x, 1};
    
    const size_t global_offset = k_local_;
    const auto k_global = k_global_;
    const auto ptr_global = global_buffer_;

    /* k_global: kernels running in the global memory */
    // if(k_global > 0)
    Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for_work_group(nb_wg_global, wg_size, [=](auto g){
            /* Solve kernel */
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it){
                    mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                    mdspan3d_t scratch_view(ptr_global, k_global, nx, ny1);

                    const int ix = it.get_local_id(1);
                    const int iy1 = g.get_group_id(2);

                    const size_t k_ny = g.get_group_id(0);
                    const int local_ny = it.get_local_id(0);
                    const int iy = wg_size_y * k_ny + ny_offset +
                                   local_ny + global_offset;

                    double const xFootCoord = displ(ix, iy, params);
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);
                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));
                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;
                    
                    scratch_view(k_ny, ix, iy1) = 0;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        scratch_view(k_ny, ix, iy1) +=
                            coef[k] * fdist_view(iy, idx_ipos1, iy1);
                    }
            }); //end parallel_for_work_item

            /* Copy kernel */
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it){
                    mdspan3d_t fdist_view(fdist.get_pointer(), ny, nx, ny1);
                    mdspan3d_t scratch_view(ptr_global, k_global, nx, ny1);

                    const int ix = it.get_local_id(1);
                    const int iy1 = g.get_group_id(2);

                    const size_t k_ny = g.get_group_id(0);
                    const int local_ny = it.get_local_id(0);
                    const int iy = wg_size_y * k_ny + ny_offset +
                                   local_ny + global_offset;

                     fdist_view(iy, ix, iy1) = scratch_view(k_ny, ix, iy1);

            }); //end parallel_for_work_item
        }); // end parallel_for_work_group
    }); //end Q.submit

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_y, nx), cgh, sycl::no_init);

        cgh.parallel_for_work_group(nb_wg_local, wg_size, [=](sycl::group<3> g) {
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

                    localAcc_view(local_ny, ix) = 0;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        localAcc_view(local_ny, ix) +=
                            coef[k] * fdist_view(iy, idx_ipos1, iy1);
                    }
                });   // end parallel_for_work_item --> Implicit barrier

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

                     fdist_view(iy, ix, iy1) = localAcc_view(local_ny, ix);
                });   // barrier
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