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
        - 40 kernels will run into global memory (3D buffer of size 40*n1*n2
            is allocated)



- Streaming in Y with blocks BY
- GridStride (Done by hierarchical) in BY and X dims
==================================================================== */

// ==========================================
// ==========================================
sycl::event
AdvX::Exp2::actual_advection(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver, const size_t &ny_batch_size,
                             const size_t &ny_offset) {

    auto const n0 = solver.p.n0;
    auto const n1 = solver.p.n1;
    auto const n2 = solver.p.n2;

    auto const wg_size_0 = solver.p.loc_wg_size_0;
    auto const wg_size_1 = solver.p.loc_wg_size_1;

    /* n0 must be divisible by slice_size_dim_y */
    if (ny_batch_size % wg_size_0 != 0) {
        throw std::invalid_argument(
            "ny_batch_size must be divisible by wg_size_0");
    }
    if (wg_size_0 * n1 > 6144) {
        throw std::invalid_argument(
            "wg_size_0*n1 must be < to 6144 (local memory limit)");
    }

    const sycl::range nb_wg_local{k_local_ / wg_size_0, 1, n2};
    const sycl::range nb_wg_global{k_global_ / wg_size_0, 1, n2};

    const sycl::range wg_size{wg_size_0, wg_size_1, 1};

    const size_t global_offset = k_local_;
    const auto k_global = k_global_;
    const auto ptr_global = global_buffer_;

    /* k_global: kernels running in the global memory */
    Q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for_work_group(nb_wg_global, wg_size, [=](auto g) {
            /* Solve kernel */
            g.parallel_for_work_item(
                sycl::range{wg_size_0, n1, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                    mdspan3d_t scratch_view(ptr_global, k_global, n1, n2);

                    const int i1 = it.get_local_id(1);
                    const int i2 = g.get_group_id(2);

                    const size_t k_ny = g.get_group_id(0);
                    const int local_ny = it.get_local_id(0);
                    const int i0 =
                        wg_size_0 * k_ny + ny_offset + local_ny + global_offset;

                    auto slice = std::experimental::submdspan(
                        fdist_view, i0, std::experimental::full_extent, i2);

                    scratch_view(k_ny, i1, i2) = solver(slice, i0, i1, i2);
                });   // end parallel_for_work_item

            /* Copy kernel */
            g.parallel_for_work_item(
                sycl::range{wg_size_0, n1, 1}, [&](sycl::h_item<3> it) {
                    mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                    mdspan3d_t scratch_view(ptr_global, k_global, n1, n2);

                    const int i1 = it.get_local_id(1);
                    const int i2 = g.get_group_id(2);

                    const size_t k_ny = g.get_group_id(0);
                    const int local_ny = it.get_local_id(0);
                    const int i0 =
                        wg_size_0 * k_ny + ny_offset + local_ny + global_offset;

                    fdist_view(i0, i1, i2) = scratch_view(k_ny, i1, i2);
                });   // end parallel_for_work_item
        });           // end parallel_for_work_group
    });               // end Q.submit

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_0, n1), cgh, sycl::no_init);

        cgh.parallel_for_work_group(
            nb_wg_local, wg_size, [=](sycl::group<3> g) {
                /* Solve kernel */
                g.parallel_for_work_item(
                    sycl::range{wg_size_0, n1, 1}, [&](sycl::h_item<3> it) {
                        mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                        mdspan2d_t localAcc_view(slice_ftmp.get_pointer(),
                                                 slice_ftmp.get_range().get(0),
                                                 slice_ftmp.get_range().get(1));

                        const int i1 = it.get_local_id(1);
                        const int i2 = g.get_group_id(2);

                        const int local_ny = it.get_local_id(0);
                        const int i0 = wg_size_0 * g.get_group_id(0) +
                                       ny_offset + local_ny;

                        auto slice = std::experimental::submdspan(
                            fdist_view, i0, std::experimental::full_extent, i2);

                        localAcc_view(local_ny, i1) = solver(slice, i0, i1, i2);
                    });   // end parallel_for_work_item --> Implicit barrier

                /* Copy kernel*/
                g.parallel_for_work_item(
                    sycl::range{wg_size_0, n1, 1}, [&](sycl::h_item<3> it) {
                        mdspan3d_t fdist_view(fdist_dev, n0, n1, n2);
                        mdspan2d_t localAcc_view(slice_ftmp.get_pointer(),
                                                 slice_ftmp.get_range().get(0),
                                                 slice_ftmp.get_range().get(1));

                        const int i1 = it.get_local_id(1);
                        const int i2 = g.get_group_id(2);

                        const int local_ny = it.get_local_id(0);
                        const int i0 = wg_size_0 * g.get_group_id(0) +
                                       ny_offset + local_ny;

                        fdist_view(i0, i1, i2) = localAcc_view(local_ny, i1);
                    });   // barrier
            });           // end parallel_for_work_group
    });                   // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Exp2::operator()(sycl::queue &Q, double *fdist_dev,
                       const Solver &solver) {

    // can be parallel on multiple streams?
    for (size_t i_batch = 0; i_batch < n_batch_ - 1; ++i_batch) {

        size_t ny_offset = (i_batch * MAX_NY_BATCHS_);

        actual_advection(Q, fdist_dev, solver, MAX_NY_BATCHS_, ny_offset).wait();
    }

    // return the last advection with the rest
    return actual_advection(Q, fdist_dev, solver, last_n0_size_,
                            last_n0_offset_);
}