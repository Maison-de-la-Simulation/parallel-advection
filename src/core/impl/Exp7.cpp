#include "advectors.h"

/* =================================================================
Coaliscing accesses + vertical distribution of data between local and
global memory: 2 types of kernels scheduled
==================================================================== */

// ==========================================
// ==========================================
sycl::event
AdvX::Exp7::actual_advection(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver, const size_t &ny_batch_size,
                             const size_t &ny_offset, const size_t k_global,
                             const size_t k_local) {

    auto const n0 = solver.p.n0;
    auto const n1 = solver.p.n1;
    auto const n2 = solver.p.n2;

    /* n0 must be divisible by slice_size_dim_y */
    if (ny_batch_size % wg_size_0 != 0) {
        throw std::invalid_argument(
            "ny_batch_size must be divisible by wg_size_0");
    }
    if (wg_size_0 * n1 > 6144) {
        throw std::invalid_argument(
            "wg_size_0*n1 must be < to 6144 (local memory limit)");
    }

    const sycl::range nb_wg_local{k_local/loc_wg_size_0_,
                                  1,
                                  n2/glob_wg_size_2_};

    const sycl::range nb_wg_global{k_global/glob_wg_size_0_,
                                   1,
                                   n2/loc_wg_size_2_};

    const sycl::range glob_wg_size{glob_wg_size_0_, glob_wg_size_1_, glob_wg_size_2_};
    const sycl::range loc_wg_size{loc_wg_size_0_  , loc_wg_size_1_, loc_wg_size_2_};

    const size_t global_offset = k_local;
    const auto ptr_global = scratchG_;

    auto const wg2 = wg_size_2_;
    auto const wg1 = wg_size_1_;

    /* k_global: kernels running in the global memory */
    auto event = Q.submit([&](sycl::handler &cgh) {
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
        sycl::local_accessor<double, 2> slice_ftmp(sycl::range<2>(wg2, n1),
                                                   cgh);

        cgh.parallel_for(sycl::nd_range<3>{global_size, local_size},
                         [=](auto itm) {
                             mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                             const int i0 = itm.get_global_id(0);
                             const int i1 = itm.get_local_id(1);
                             const int i2 = itm.get_global_id(2);

                             auto slice = std::experimental::submdspan(
                                 fdist, i0, std::experimental::full_extent, i2);

                             for (int ii1 = i1; ii1 < n1; ii1 += wg1) {
                                 slice_ftmp[i2][ii1] =
                                     solver(slice, i0, ii1, i2);
                             }
                             // }

                             sycl::group_barrier(itm.get_group());

                             for (int ii1 = i1; ii1 < n1; ii1 += wg1) {
                                 fdist(i0, ii1, i2) = slice_ftmp[i2][ii1];
                             }
                         }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit
}   // end actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::Alg5::operator()(sycl::queue &Q, double *fdist_dev,
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